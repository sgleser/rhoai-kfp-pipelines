#!/usr/bin/env python3
"""
Kubeflow Pipeline for deploying and testing models with KServe + vLLM
Uses KServe InferenceService for persistent model serving
"""

import kfp
from kfp import dsl
from kfp import compiler
from kfp import kubernetes

# Custom image for pipeline components
BASE_IMAGE = "quay.io/rh_ee_sgleszer/vllm-model-test:0.0.3"


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["kubernetes"]
)
def download_model(
    model_name: str,
    cache_dir: str = "/mnt/models"
) -> str:
    """
    Downloads a model from HuggingFace Hub to shared storage
    """
    from huggingface_hub import snapshot_download
    import os
    
    print(f"Downloading model: {model_name}")
    print(f"Cache directory: {cache_dir}")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    model_path = snapshot_download(
        repo_id=model_name,
        cache_dir=cache_dir,
        resume_download=True
    )
    
    print(f"Model downloaded successfully to: {model_path}")
    return model_name


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["kubernetes"]
)
def deploy_kserve_model(
    model_name: str,
    namespace: str,
    pvc_name: str,
    inference_service_name: str,
    runtime: str = "vllm",
    gpu_count: int = 1,
    max_model_len: int = 2048
) -> str:
    """
    Deploys a model using KServe InferenceService with vLLM runtime
    
    Args:
        model_name: HuggingFace model ID
        namespace: Kubernetes namespace for the InferenceService
        pvc_name: PVC containing the downloaded model
        inference_service_name: Name for the InferenceService
        runtime: Serving runtime (vllm)
        gpu_count: Number of GPUs to allocate
        max_model_len: Maximum sequence length for vLLM
    
    Returns:
        The InferenceService endpoint URL
    """
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException
    import time
    import os
    
    print(f"Deploying model {model_name} with KServe")
    print(f"Namespace: {namespace}")
    print(f"InferenceService name: {inference_service_name}")
    
    # Load in-cluster config
    config.load_incluster_config()
    
    # Create custom objects API for KServe resources
    api = client.CustomObjectsApi()
    
    # Convert model name to path format (e.g., facebook/opt-125m -> facebook--opt-125m)
    model_path = model_name.replace("/", "--")
    
    # InferenceService manifest for vLLM runtime
    # This uses the OpenShift AI / RHOAI vLLM ServingRuntime
    inference_service = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": inference_service_name,
            "namespace": namespace,
            "annotations": {
                "serving.kserve.io/deploymentMode": "RawDeployment"
            }
        },
        "spec": {
            "predictor": {
                "model": {
                    "modelFormat": {
                        "name": "vLLM"
                    },
                    "runtime": "vllm-runtime",
                    "storageUri": f"pvc://{pvc_name}/models--{model_path}",
                    "args": [
                        "--max-model-len", str(max_model_len),
                        "--dtype", "auto"
                    ]
                },
                "resources": {
                    "limits": {
                        "nvidia.com/gpu": str(gpu_count),
                        "memory": "16Gi",
                        "cpu": "8"
                    },
                    "requests": {
                        "nvidia.com/gpu": str(gpu_count),
                        "memory": "8Gi",
                        "cpu": "4"
                    }
                },
                "tolerations": [
                    {
                        "key": "nvidia.com/gpu",
                        "operator": "Equal",
                        "value": "Tesla-T4-SHARED",
                        "effect": "NoSchedule"
                    }
                ]
            }
        }
    }
    
    # Check if InferenceService already exists
    try:
        existing = api.get_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            name=inference_service_name
        )
        print(f"InferenceService {inference_service_name} already exists, deleting...")
        api.delete_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            name=inference_service_name
        )
        # Wait for deletion
        time.sleep(10)
    except ApiException as e:
        if e.status != 404:
            raise
        print(f"InferenceService {inference_service_name} does not exist, creating...")
    
    # Create the InferenceService
    api.create_namespaced_custom_object(
        group="serving.kserve.io",
        version="v1beta1",
        namespace=namespace,
        plural="inferenceservices",
        body=inference_service
    )
    
    print(f"InferenceService {inference_service_name} created, waiting for ready state...")
    
    # Wait for the InferenceService to be ready
    max_wait_time = 600  # 10 minutes
    poll_interval = 10
    elapsed = 0
    endpoint_url = None
    
    while elapsed < max_wait_time:
        time.sleep(poll_interval)
        elapsed += poll_interval
        
        try:
            isvc = api.get_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=namespace,
                plural="inferenceservices",
                name=inference_service_name
            )
            
            status = isvc.get("status", {})
            conditions = status.get("conditions", [])
            
            # Check if Ready condition is True
            for condition in conditions:
                if condition.get("type") == "Ready":
                    if condition.get("status") == "True":
                        # Get the URL
                        endpoint_url = status.get("url", "")
                        if not endpoint_url:
                            # Try to construct from address
                            address = status.get("address", {})
                            endpoint_url = address.get("url", "")
                        
                        print(f"[OK] InferenceService is ready!")
                        print(f"Endpoint URL: {endpoint_url}")
                        return endpoint_url
                    else:
                        reason = condition.get("reason", "Unknown")
                        message = condition.get("message", "")
                        print(f"Waiting... Status: {reason} - {message} ({elapsed}s)")
                        break
        except ApiException as e:
            print(f"Error checking status: {e}")
        
        print(f"Waiting for InferenceService... ({elapsed}s)")
    
    raise RuntimeError(f"InferenceService did not become ready within {max_wait_time}s")


@dsl.component(
    base_image=BASE_IMAGE
)
def test_kserve_endpoint(
    endpoint_url: str,
    model_name: str,
    test_prompt: str = "Once upon a time",
    run_promptfoo: bool = False
) -> str:
    """
    Tests the KServe vLLM endpoint with inference requests
    
    Args:
        endpoint_url: The KServe InferenceService URL
        model_name: Model name for the API
        test_prompt: Prompt to test with
        run_promptfoo: Whether to run Promptfoo-style evaluation tests
    
    Returns:
        Test results summary
    """
    import json
    import urllib.request
    import urllib.error
    import time
    
    print(f"Testing KServe endpoint: {endpoint_url}")
    print(f"Model: {model_name}")
    
    # Construct the completions endpoint
    # KServe vLLM exposes OpenAI-compatible API at /v1/completions
    if endpoint_url.endswith("/"):
        endpoint_url = endpoint_url.rstrip("/")
    
    completions_url = f"{endpoint_url}/v1/completions"
    
    # Wait a bit for the endpoint to be fully ready
    print("Waiting for endpoint to be fully ready...")
    time.sleep(10)
    
    # Run inference test
    print("\n" + "="*60)
    print("RUNNING INFERENCE TEST")
    print("="*60)
    
    test_payload = {
        "model": model_name,
        "prompt": test_prompt,
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    max_retries = 5
    test_result = None
    
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                completions_url,
                data=json.dumps(test_payload).encode('utf-8'),
                headers={"Content-Type": "application/json"}
            )
            response = urllib.request.urlopen(req, timeout=120)
            result = json.loads(response.read().decode('utf-8'))
            
            generated_text = result['choices'][0]['text']
            
            print(f"\nPrompt: {test_prompt}")
            print(f"\nResponse: {generated_text}")
            print(f"\n[OK] Inference test PASSED!")
            
            test_result = f"SUCCESS: Generated {len(generated_text)} characters"
            break
            
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(10)
            else:
                test_result = f"FAILED: {str(e)}"
    
    # Run Promptfoo-style evaluation tests if enabled
    if run_promptfoo and test_result and "SUCCESS" in test_result:
        print("\n" + "="*60)
        print("RUNNING PROMPTFOO-STYLE EVALUATION TESTS")
        print("="*60)
        
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
                    completions_url,
                    data=json.dumps(payload).encode('utf-8'),
                    headers={"Content-Type": "application/json"}
                )
                response = urllib.request.urlopen(req, timeout=120)
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
        
        test_result = f"{test_result} | Promptfoo: {promptfoo_results['passed']}/{promptfoo_results['passed'] + promptfoo_results['failed']} passed"
    
    print("\n" + "="*60)
    print(f"FINAL RESULT: {test_result}")
    print("="*60)
    
    return test_result


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=["kubernetes"]
)
def cleanup_kserve_model(
    inference_service_name: str,
    namespace: str,
    delete_service: bool = False
) -> str:
    """
    Optionally cleans up the KServe InferenceService after testing
    
    Args:
        inference_service_name: Name of the InferenceService
        namespace: Kubernetes namespace
        delete_service: Whether to actually delete (False = keep running)
    
    Returns:
        Status message
    """
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException
    
    if not delete_service:
        print(f"Keeping InferenceService {inference_service_name} running")
        print(f"You can access it at: https://{inference_service_name}-{namespace}.apps.<cluster-domain>/v1/completions")
        return f"InferenceService {inference_service_name} kept running"
    
    print(f"Deleting InferenceService {inference_service_name}")
    
    config.load_incluster_config()
    api = client.CustomObjectsApi()
    
    try:
        api.delete_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            name=inference_service_name
        )
        print(f"[OK] InferenceService {inference_service_name} deleted")
        return f"InferenceService {inference_service_name} deleted successfully"
    except ApiException as e:
        if e.status == 404:
            print(f"InferenceService {inference_service_name} not found (already deleted?)")
            return f"InferenceService {inference_service_name} not found"
        raise


@dsl.pipeline(
    name="kserve-vllm-model-test",
    description="Downloads a model, deploys it with KServe vLLM, runs tests, optionally keeps the service running"
)
def kserve_model_pipeline(
    model_name: str = "facebook/opt-125m",
    namespace: str = "engagement-test",
    pvc_name: str = "model-storage-pvc",
    inference_service_name: str = "vllm-test-model",
    cache_dir: str = "/mnt/models",
    gpu_count: int = 1,
    max_model_len: int = 2048,
    test_prompt: str = "Once upon a time",
    run_promptfoo: bool = True,
    cleanup_after_test: bool = False  # Set to True to delete InferenceService after test
):
    """
    Pipeline to download, deploy with KServe, and test models
    
    The InferenceService remains running after the pipeline completes,
    allowing you to access it from other containers or applications.
    
    Args:
        model_name: HuggingFace model ID
        namespace: Kubernetes namespace
        pvc_name: PVC for model storage
        inference_service_name: Name for the KServe InferenceService
        cache_dir: Directory for model cache
        gpu_count: Number of GPUs
        max_model_len: Maximum sequence length
        test_prompt: Prompt for testing
        run_promptfoo: Run Promptfoo-style tests
        cleanup_after_test: Delete InferenceService after testing (default: keep running)
    """
    # Step 1: Download the model
    download_task = download_model(
        model_name=model_name,
        cache_dir=cache_dir
    )
    
    kubernetes.mount_pvc(
        download_task,
        pvc_name=pvc_name,
        mount_path="/mnt/models"
    )
    
    download_task.set_cpu_request("2")
    download_task.set_cpu_limit("4")
    download_task.set_memory_request("4Gi")
    download_task.set_memory_limit("8Gi")
    
    # Step 2: Deploy with KServe
    deploy_task = deploy_kserve_model(
        model_name=download_task.output,
        namespace=namespace,
        pvc_name=pvc_name,
        inference_service_name=inference_service_name,
        gpu_count=gpu_count,
        max_model_len=max_model_len
    )
    
    deploy_task.set_cpu_request("1")
    deploy_task.set_cpu_limit("2")
    deploy_task.set_memory_request("512Mi")
    deploy_task.set_memory_limit("1Gi")
    
    # Step 3: Test the endpoint
    test_task = test_kserve_endpoint(
        endpoint_url=deploy_task.output,
        model_name=model_name,
        test_prompt=test_prompt,
        run_promptfoo=run_promptfoo
    )
    
    test_task.set_cpu_request("1")
    test_task.set_cpu_limit("2")
    test_task.set_memory_request("512Mi")
    test_task.set_memory_limit("1Gi")
    
    # Step 4: Cleanup (optional)
    cleanup_task = cleanup_kserve_model(
        inference_service_name=inference_service_name,
        namespace=namespace,
        delete_service=cleanup_after_test
    )
    cleanup_task.after(test_task)
    
    cleanup_task.set_cpu_request("500m")
    cleanup_task.set_cpu_limit("1")
    cleanup_task.set_memory_request("256Mi")
    cleanup_task.set_memory_limit("512Mi")


if __name__ == '__main__':
    import subprocess
    import os

    # Pipeline arguments
    arguments = {
        "model_name": "facebook/opt-125m",
        "namespace": "engagement-test",
        "pvc_name": "model-storage-pvc",
        "inference_service_name": "vllm-opt-125m",
        "cache_dir": "/mnt/models",
        "gpu_count": 1,
        "max_model_len": 2048,
        "test_prompt": "Once upon a time",
        "run_promptfoo": True,
        "cleanup_after_test": False,  # Keep the service running!
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

    print(f'Submitting KServe pipeline with model: {arguments["model_name"]}')
    print(f'InferenceService name: {arguments["inference_service_name"]}')
    print(f'Cleanup after test: {arguments["cleanup_after_test"]}')
    
    run = client.create_run_from_pipeline_func(
        kserve_model_pipeline,
        arguments=arguments,
        experiment_name="kserve-vllm-test",
        enable_caching=False
    )
    print(f'Pipeline run submitted successfully!')
    print(f'Run ID: {run.run_id}')
    print(f'\nAfter the pipeline completes, the InferenceService will be available at:')
    print(f'https://{arguments["inference_service_name"]}-{arguments["namespace"]}.apps.prod.rhoai.rh-aiservices-bu.com/v1/completions')

