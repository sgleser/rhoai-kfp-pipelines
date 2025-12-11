#!/usr/bin/env python3
"""
Kubeflow Pipeline for downloading and testing models with vLLM server
Uses PersistentVolumeClaim for shared storage between components
"""

import kfp
from kfp import dsl
from kfp import compiler
from kfp import kubernetes

# Custom image for download component (based on Red Hat UBI 9 Python 3.11)
BASE_IMAGE = "quay.io/rh_ee_sgleszer/vllm-model-test:0.0.3"

# Official vLLM image with pre-built CUDA kernels (required for GPU inference)
# VLLM_IMAGE = "vllm/vllm-openai:latest"


@dsl.component(
    base_image=BASE_IMAGE
)
def download_model(
    model_name: str,
    cache_dir: str = "/mnt/models"
) -> str:
    """
    Downloads a model from HuggingFace Hub to shared storage
    
    Args:
        model_name: HuggingFace model ID (e.g., "facebook/opt-125m")
        cache_dir: Directory to cache the downloaded model (must be PVC-mounted)
    
    Returns:
        Path to the downloaded model
    """
    from huggingface_hub import snapshot_download
    import os
    
    print(f"Downloading model: {model_name}")
    print(f"Cache directory: {cache_dir}")
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Download the model
    model_path = snapshot_download(
        repo_id=model_name,
        cache_dir=cache_dir,
        resume_download=True
    )
    
    print(f"Model downloaded successfully to: {model_path}")
    
    # Return just the model name for the next component
    # The actual files are on the shared PVC
    return model_name


@dsl.component(
    base_image=BASE_IMAGE
)
def start_vllm_and_test(
    model_name: str,
    test_prompt: str = "Once upon a time",
    cache_dir: str = "/mnt/models",
    port: int = 8000,
    tensor_parallel_size: int = 1,
    max_model_len: int = 2048,
    run_promptfoo: bool = False
) -> str:
    """
    Starts a vLLM server, runs inference tests, then exits.
    Server and tests run in the same pod so they can communicate.
    
    Args:
        model_name: HuggingFace model ID
        test_prompt: Prompt to test the model with
        cache_dir: Directory where model is cached (must be PVC-mounted)
        port: Port to run the server on
        tensor_parallel_size: Number of GPUs to use
        max_model_len: Maximum sequence length
        run_promptfoo: Whether to run Promptfoo-style evaluation tests
    
    Returns:
        Test results summary
    """
    import subprocess
    import time
    import os
    import json
    import urllib.request
    import urllib.error
    
    print(f"Starting vLLM server for model: {model_name}")
    print(f"Cache directory: {cache_dir}")
    print(f"Port: {port}")
    
    # Set HuggingFace cache environment variables
    os.environ['HF_HOME'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    
    # Fix permission issues - redirect cache directories to writable locations
    # Set HOME to /tmp so all .cache directories go there
    os.environ['HOME'] = '/tmp'
    os.environ['XDG_CACHE_HOME'] = '/tmp/.cache'
    os.environ['VLLM_CACHE_ROOT'] = '/tmp/.cache/vllm'
    os.environ['FLASHINFER_WORKSPACE_DIR'] = '/tmp/.cache/flashinfer'
    
    # Use Triton attention backend instead of FlashInfer (which requires nvcc for JIT)
    os.environ['VLLM_ATTENTION_BACKEND'] = 'TRITON_ATTN'
    
    # Create cache directories
    os.makedirs('/tmp/.cache/vllm', exist_ok=True)
    os.makedirs('/tmp/.cache/flashinfer', exist_ok=True)

    # Build vLLM command
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--max-model-len", str(max_model_len)
    ]

    print(f"Running command: {' '.join(cmd)}")
    print(f"Environment: HOME={os.environ.get('HOME')}, FLASHINFER_WORKSPACE_DIR={os.environ.get('FLASHINFER_WORKSPACE_DIR')}")

    # Start the server in background - pass full environment including our cache fixes
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=os.environ.copy()  # Explicitly pass the modified environment
    )
    
    # Wait for server to be ready
    print("Waiting for vLLM server to start...")
    server_url = f"http://localhost:{port}"
    max_retries = 60  # 5 minutes max wait
    server_ready = False
    
    for i in range(max_retries):
        time.sleep(5)
        try:
            req = urllib.request.Request(f"{server_url}/health")
            urllib.request.urlopen(req, timeout=5)
            print(f"[OK] Server is ready after {(i+1)*5} seconds!")
            server_ready = True
            break
        except Exception as e:
            if process.poll() is not None:
                # Process died, get output
                output, _ = process.communicate()
                print(f"[ERROR] Server process died! Output:\n{output.decode()}")
                raise RuntimeError("vLLM server failed to start")
            print(f"Waiting for server... ({(i+1)*5}s)")
    
    if not server_ready:
        process.terminate()
        raise RuntimeError("Server failed to start within timeout")
    
    # Run inference test using OpenAI-compatible API
    # Use /v1/completions for completion models (OPT, GPT-2, etc.)
    # Use /v1/chat/completions for chat models (Llama-chat, etc.)
    print("\n" + "="*60)
    print("RUNNING INFERENCE TEST")
    print("="*60)

    # Use completions endpoint (works with all models including non-chat models)
    test_payload = {
        "model": model_name,
        "prompt": test_prompt,
        "max_tokens": 100,
        "temperature": 0.7
    }

    try:
        req = urllib.request.Request(
            f"{server_url}/v1/completions",
            data=json.dumps(test_payload).encode('utf-8'),
            headers={"Content-Type": "application/json"}
        )
        response = urllib.request.urlopen(req, timeout=60)
        result = json.loads(response.read().decode('utf-8'))

        generated_text = result['choices'][0]['text']

        print(f"\nPrompt: {test_prompt}")
        print(f"\nResponse: {generated_text}")
        print(f"\n[OK] Inference test PASSED!")

        test_result = f"SUCCESS: Generated {len(generated_text)} characters"

    except Exception as e:
        print(f"[ERROR] Inference test failed: {e}")
        test_result = f"FAILED: {str(e)}"

    # Run Promptfoo-style evaluation tests if enabled
    if run_promptfoo and "SUCCESS" in test_result:
        print("\n" + "="*60)
        print("RUNNING PROMPTFOO-STYLE EVALUATION TESTS")
        print("="*60)
        
        # Define test cases similar to Promptfoo config
        test_cases = [
            {
                "description": "Story generation test",
                "prompt": "Write a short story about a robot learning to paint",
                "assertions": [
                    {"type": "min_length", "value": 50},
                    {"type": "max_length", "value": 500},
                ]
            },
            {
                "description": "Explanation test",
                "prompt": "Explain machine learning in simple terms",
                "assertions": [
                    {"type": "min_length", "value": 30},
                ]
            },
            {
                "description": "Code generation test",
                "prompt": "Write Python code to sort a list:",
                "assertions": [
                    {"type": "contains_any", "value": ["sort", "sorted", "def", "list"]},
                ]
            },
        ]
        
        promptfoo_results = {"passed": 0, "failed": 0, "tests": []}
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- Test {i}: {test_case['description']} ---")
            print(f"Prompt: {test_case['prompt'][:50]}...")
            
            try:
                payload = {
                    "model": model_name,
                    "prompt": test_case["prompt"],
                    "max_tokens": 150,
                    "temperature": 0.7
                }
                
                req = urllib.request.Request(
                    f"{server_url}/v1/completions",
                    data=json.dumps(payload).encode('utf-8'),
                    headers={"Content-Type": "application/json"}
                )
                response = urllib.request.urlopen(req, timeout=60)
                result = json.loads(response.read().decode('utf-8'))
                output = result['choices'][0]['text']
                
                # Run assertions
                test_passed = True
                assertion_results = []
                
                for assertion in test_case["assertions"]:
                    if assertion["type"] == "min_length":
                        passed = len(output) >= assertion["value"]
                        assertion_results.append(f"min_length({assertion['value']}): {'PASS' if passed else 'FAIL'}")
                    elif assertion["type"] == "max_length":
                        passed = len(output) <= assertion["value"]
                        assertion_results.append(f"max_length({assertion['value']}): {'PASS' if passed else 'FAIL'}")
                    elif assertion["type"] == "contains_any":
                        passed = any(word.lower() in output.lower() for word in assertion["value"])
                        assertion_results.append(f"contains_any({assertion['value']}): {'PASS' if passed else 'FAIL'}")
                    
                    if not passed:
                        test_passed = False
                
                status = "PASS" if test_passed else "FAIL"
                print(f"Output length: {len(output)} chars")
                print(f"Assertions: {', '.join(assertion_results)}")
                print(f"Result: [{status}]")
                
                if test_passed:
                    promptfoo_results["passed"] += 1
                else:
                    promptfoo_results["failed"] += 1
                    
                promptfoo_results["tests"].append({
                    "description": test_case["description"],
                    "status": status,
                    "output_length": len(output)
                })
                
            except Exception as e:
                print(f"[ERROR] Test failed: {e}")
                promptfoo_results["failed"] += 1
                promptfoo_results["tests"].append({
                    "description": test_case["description"],
                    "status": "ERROR",
                    "error": str(e)
                })
        
        print("\n" + "="*60)
        print("PROMPTFOO EVALUATION SUMMARY")
        print("="*60)
        print(f"Total tests: {promptfoo_results['passed'] + promptfoo_results['failed']}")
        print(f"Passed: {promptfoo_results['passed']}")
        print(f"Failed: {promptfoo_results['failed']}")
        
        # Update test result with Promptfoo results
        test_result = f"{test_result} | Promptfoo: {promptfoo_results['passed']}/{promptfoo_results['passed'] + promptfoo_results['failed']} passed"

    # Cleanup - terminate server
    print("\n" + "="*60)
    print("SHUTTING DOWN SERVER")
    print("="*60)
    process.terminate()
    process.wait(timeout=10)
    print("[OK] Server stopped successfully")
    
    print("\n" + "="*60)
    print(f"FINAL RESULT: {test_result}")
    print("="*60)
    
    return test_result


@dsl.component(
    base_image="registry.access.redhat.com/ubi9/nodejs-20:latest",
    packages_to_install=[]
)
def test_with_promptfoo(
    server_url: str,
    model_name: str,
    config_dir: str = "/mnt/models/promptfoo"
):
    """
    Tests the vLLM server using Promptfoo evaluation framework
    
    Args:
        server_url: URL of the vLLM server
        model_name: Model name to use in tests
        config_dir: Directory to store promptfoo config and results (on PVC)
    """
    import subprocess
    import json
    import os
    import time
    
    print(f"Installing Promptfoo...")
    subprocess.run(["npm", "install", "-g", "promptfoo"], check=True)
    
    print(f"Testing vLLM server at: {server_url}")
    print(f"Model: {model_name}")
    
    # Create config directory
    os.makedirs(config_dir, exist_ok=True)
    os.chdir(config_dir)
    
    # Wait for server to be ready
    print("Waiting for vLLM server to be ready...")
    max_retries = 20
    for i in range(max_retries):
        try:
            import urllib.request
            urllib.request.urlopen(f"{server_url}/health", timeout=5)
            print("Server is ready!")
            break
        except Exception as e:
            print(f"Waiting... (attempt {i+1}/{max_retries})")
            time.sleep(10)
    else:
        print("WARNING: Server health check failed, proceeding anyway...")
    
    # Create promptfoo configuration
    config = {
        "description": f"vLLM Model Test - {model_name}",
        "providers": [
            {
                "id": "openai:chat:gpt-3.5-turbo",
                "config": {
                    "apiBaseUrl": f"{server_url}/v1",
                    "apiKey": "dummy-key",
                }
            }
        ],
        "prompts": [
            "Write a short story about {{topic}}",
            "Explain {{concept}} in simple terms",
            "Generate code to {{task}}"
        ],
        "tests": [
            {
                "vars": {
                    "topic": "a robot learning to paint"
                },
                "assert": [
                    {
                        "type": "contains",
                        "value": "robot"
                    },
                    {
                        "type": "javascript",
                        "value": "output.length > 50"
                    }
                ]
            },
            {
                "vars": {
                    "concept": "machine learning"
                },
                "assert": [
                    {
                        "type": "contains-any",
                        "value": ["model", "data", "learning", "algorithm"]
                    },
                    {
                        "type": "javascript",
                        "value": "output.length > 30"
                    }
                ]
            },
            {
                "vars": {
                    "task": "sort a list in Python"
                },
                "assert": [
                    {
                        "type": "contains-any",
                        "value": ["sort", "sorted", "python"]
                    },
                    {
                        "type": "javascript",
                        "value": "output.length > 20"
                    }
                ]
            },
            {
                "description": "Factual accuracy test",
                "vars": {
                    "topic": "the solar system"
                },
                "assert": [
                    {
                        "type": "llm-rubric",
                        "value": "The output should mention planets and be factually accurate"
                    }
                ]
            },
            {
                "description": "Coherence test",
                "vars": {
                    "concept": "quantum computing"
                },
                "assert": [
                    {
                        "type": "javascript",
                        "value": "output.length > 50 && output.length < 500"
                    }
                ]
            }
        ],
        "outputPath": "./promptfoo_results.json"
    }
    
    # Write config file
    config_file = "promptfooconfig.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Created Promptfoo config: {config_file}")
    print(json.dumps(config, indent=2))
    
    # Run promptfoo evaluation
    print("\n" + "="*60)
    print("Running Promptfoo evaluation...")
    print("="*60 + "\n")
    
    try:
        result = subprocess.run(
            ["promptfoo", "eval", "-c", config_file],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings/Info:", result.stderr)
        
        # Display results
        print("\n" + "="*60)
        print("Generating report...")
        print("="*60 + "\n")
        
        subprocess.run(
            ["promptfoo", "view", "--no-open"],
            check=False
        )
        
        # Print summary
        if os.path.exists("promptfoo_results.json"):
            with open("promptfoo_results.json", "r") as f:
                results = json.load(f)
            print("\n" + "="*60)
            print("EVALUATION SUMMARY")
            print("="*60)
            print(json.dumps(results.get("stats", {}), indent=2))
        
        print("\nPromptfoo evaluation completed successfully!")
        print(f"Results saved to: {config_dir}")
        
    except subprocess.CalledProcessError as e:
        print(f"Promptfoo evaluation failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise


@dsl.component(
    base_image=BASE_IMAGE
)
def test_vllm_server(
    server_url: str,
    model_name: str,
    test_prompt: str = "Hello, how are you?"
):
    """
    Simple health check and basic test of the vLLM server
    
    Args:
        server_url: URL of the vLLM server
        model_name: Model name to use in the request
        test_prompt: Test prompt to send
    """
    import requests
    import json
    import time
    
    print(f"Testing vLLM server at: {server_url}")
    print(f"Test prompt: {test_prompt}")
    
    # Wait for server to be ready
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get(f"{server_url}/health")
            if response.status_code == 200:
                print("Server is healthy!")
                break
        except Exception as e:
            print(f"Waiting for server... (attempt {i+1}/{max_retries})")
            time.sleep(5)
    else:
        print("Server health check failed")
        return
    
    # Test completion endpoint
    try:
        payload = {
            "model": model_name,
            "prompt": test_prompt,
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        print(f"Sending request: {json.dumps(payload, indent=2)}")
        
        response = requests.post(
            f"{server_url}/v1/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("Success! Response:")
            print(json.dumps(result, indent=2))
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Error testing server: {e}")
        raise


@dsl.pipeline(
    name="model-download-and-vllm-test",
    description="Downloads a model from HuggingFace and starts a vLLM server for testing with PVC storage and Promptfoo evaluation"
)
def model_test_pipeline(
    model_name: str = "facebook/opt-125m",  # Small model for testing
    pvc_name: str = "model-storage-pvc",  # PVC for persistent model storage
    cache_dir: str = "/mnt/models",
    server_port: int = 8000,
    tensor_parallel_size: int = 1,
    max_model_len: int = 2048,
    run_promptfoo: bool = False,  # Enable Promptfoo testing
    test_prompt: str = "Once upon a time",
    # Download component resources
    download_cpu_request: str = "2",
    download_cpu_limit: str = "4",
    download_memory_request: str = "4Gi",
    download_memory_limit: str = "8Gi",
    # vLLM server component resources
    vllm_cpu_request: str = "4",
    vllm_cpu_limit: str = "8",
    vllm_memory_request: str = "8Gi",
    vllm_memory_limit: str = "16Gi"
):
    """
    Pipeline to download and test models with vLLM using PVC for shared storage
    
    Args:
        model_name: HuggingFace model ID
        pvc_name: Name of the PersistentVolumeClaim to use for model storage
        cache_dir: Directory to cache models (mount point for PVC)
        server_port: Port for vLLM server
        tensor_parallel_size: Number of GPUs
        max_model_len: Maximum sequence length
        run_promptfoo: Enable automated testing with Promptfoo
        test_prompt: Prompt to test the model (for basic test)
        download_cpu_request: CPU request for download component
        download_cpu_limit: CPU limit for download component
        download_memory_request: Memory request for download component (e.g., "4Gi")
        download_memory_limit: Memory limit for download component (e.g., "8Gi")
        vllm_cpu_request: CPU request for vLLM server
        vllm_cpu_limit: CPU limit for vLLM server
        vllm_memory_request: Memory request for vLLM server (e.g., "8Gi")
        vllm_memory_limit: Memory limit for vLLM server (e.g., "16Gi")
    """
    # Step 1: Download the model
    download_task = download_model(
        model_name=model_name,
        cache_dir=cache_dir
    )
    
    # Mount PVC to download task (mount_path must be a constant string)
    kubernetes.mount_pvc(
        download_task,
        pvc_name=pvc_name,
        mount_path="/mnt/models"
    )
    
    # Set resource requirements for download
    # Note: resource values must be constants for older kfp versions
    download_task.set_cpu_request("2")
    download_task.set_cpu_limit("4")
    download_task.set_memory_request("4Gi")
    download_task.set_memory_limit("8Gi")
    
    # Step 2: Start vLLM server and run inference test (combined in one pod)
    vllm_task = start_vllm_and_test(
        model_name=download_task.output,
        test_prompt=test_prompt,
        cache_dir=cache_dir,
        port=server_port,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        run_promptfoo=run_promptfoo
    )
    
    # Mount the same PVC to vLLM task
    kubernetes.mount_pvc(
        vllm_task,
        pvc_name=pvc_name,
        mount_path="/mnt/models"
    )
    
    # Add toleration for GPU node
    kubernetes.add_toleration(
        vllm_task,
        key="nvidia.com/gpu",
        operator="Equal",
        value="Tesla-T4-SHARED",
        effect="NoSchedule"
    )
    
    # Set GPU and resource requirements for vLLM server
    vllm_task.set_accelerator_type("nvidia.com/gpu")
    vllm_task.set_accelerator_limit(1)
    vllm_task.set_cpu_request("4")
    vllm_task.set_cpu_limit("8")
    vllm_task.set_memory_request("8Gi")
    vllm_task.set_memory_limit("16Gi")



if __name__ == '__main__':
    import subprocess
    import os

    # Pipeline arguments
    arguments = {
        "model_name": "facebook/opt-125m",  # Small model for testing
        "pvc_name": "model-storage-pvc",
        "cache_dir": "/mnt/models",
        "server_port": 8000,
        "tensor_parallel_size": 1,
        "max_model_len": 2048,
        "test_prompt": "Once upon a time",
    }

    # Data Science Pipelines route
    kubeflow_endpoint = "https://ds-pipeline-dspa-engagement-test.apps.prod.rhoai.rh-aiservices-bu.com"

    # Get token from oc login
    try:
        bearer_token = subprocess.check_output(
            ["oc", "whoami", "-t"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except subprocess.CalledProcessError:
        print("Error: Could not get token. Please run 'oc login' first.")
        exit(1)

    print(f'Connecting to Data Science Pipelines: {kubeflow_endpoint}')
    client = kfp.Client(
        host=kubeflow_endpoint,
        existing_token=bearer_token,
    )

    print(f'Submitting pipeline run with model: {arguments["model_name"]}')
    run = client.create_run_from_pipeline_func(
        model_test_pipeline,
        arguments=arguments,
        experiment_name="vllm-model-test",
        enable_caching=False
    )
    print(f'Pipeline run submitted successfully!')
    print(f'Run ID: {run.run_id}')