#!/usr/bin/env python3
"""
Kubeflow Pipeline for evaluating LLMs using lm-eval harness with vLLM backend.
Loads model directly in the pod (requires GPU).
Uses German language tasks (XNLI-de, MLQA-de).
"""

from kfp import dsl
import kfp


@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
    packages_to_install=[
        "lm-eval[vllm]",
        "unitxt",
        "transformers",
        "torch",
        "accelerate",
    ],
)
def evaluate_model(
    model_path: str,
    output_metrics: dsl.Output[dsl.Metrics],
    output_results: dsl.Output[dsl.Artifact],
    limit: int = 100,
    batch_size: int = 1,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.8,
    dtype: str = "auto",
):
    """
    Evaluates an LLM using lm-eval harness with vLLM backend (loads model on GPU).
    
    Args:
        model_path: HuggingFace model ID or local path
        limit: Max examples per task
        batch_size: Batch size for evaluation
        max_model_len: Maximum model context length
        gpu_memory_utilization: GPU memory utilization (0.0-1.0)
        dtype: Data type (auto, float16, bfloat16)
    """
    import os
    import json
    import logging
    import time

    import torch
    from lm_eval.api.registry import get_model
    from lm_eval.evaluator import evaluate
    from lm_eval.tasks import get_task_dict
    from lm_eval.tasks.unitxt import task

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    print("=" * 60)
    print("LM-EVAL BENCHMARK (vLLM Backend - German)")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Limit: {limit}")
    print(f"Batch size: {batch_size}")
    print(f"Max model len: {max_model_len}")
    print(f"GPU memory: {gpu_memory_utilization}")
    print(f"Dtype: {dtype}")
    print("=" * 60)

    # Validate GPU
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. This pipeline requires a GPU.")
    
    logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")

    # Define Unitxt tasks (German)
    UNITXT_TASKS = [
        {
            "task": "xnli_german",
            "recipe": "card=cards.xnli.de,template=templates.classification.multi_class.relation.default",
            "group": "classification",
            "output_type": "generate_until",
        },
        {
            "task": "mlqa_german",
            "recipe": "card=cards.mlqa.de,template=templates.qa.with_context.simple",
            "group": "qa",
            "output_type": "generate_until",
        },
    ]

    # Create Unitxt task objects
    logger.info("Creating Unitxt tasks...")
    eval_tasks = []
    for config in UNITXT_TASKS:
        task_obj = task.Unitxt(config=config)
        task_obj.config.task = config["task"]
        eval_tasks.append(task_obj)
        logger.info(f"  Created task: {config['task']}")

    task_dict = get_task_dict(eval_tasks)
    logger.info(f"Created {len(task_dict)} tasks")

    # Load model with vLLM backend
    logger.info("Loading model with vLLM backend...")
    start_time = time.time()

    model_args = {
        "pretrained": model_path,
        "dtype": dtype,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": True,
        "add_bos_token": True,
    }

    model_class = get_model("vllm")
    additional_config = {
        "batch_size": batch_size,
        "max_batch_size": None,
        "device": None,
    }

    loaded_model = model_class.create_from_arg_obj(model_args, additional_config)
    logger.info(f"Model loaded in {time.time() - start_time:.2f}s")

    # Run evaluation
    logger.info("Running evaluation...")
    start_time = time.time()

    results = evaluate(
        lm=loaded_model,
        task_dict=task_dict,
        limit=limit,
    )

    logger.info(f"Evaluation completed in {time.time() - start_time:.2f}s")

    # Clean for JSON
    def clean_json(obj):
        if isinstance(obj, dict):
            return {k: clean_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_json(i) for i in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        return str(obj)

    clean_results = clean_json(results)

    # Save results
    output_results.name = "results.json"
    with open(output_results.path, "w") as f:
        json.dump(clean_results, f, indent=2)

    # Log metrics
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for task_name, task_results in clean_results.get("results", {}).items():
        print(f"\n{task_name}:")
        for metric, value in task_results.items():
            if isinstance(value, (int, float)):
                if "acc" in metric.lower():
                    print(f"  {metric}: {value:.2%}")
                else:
                    print(f"  {metric}: {value:.4f}")
                if value != 0:
                    output_metrics.log_metric(f"{task_name}_{metric}", value)

    print("\n" + "=" * 60)


@dsl.pipeline(
    name="lm-eval-vllm-german",
    description="Evaluate LLM using lm-eval with vLLM backend (GPU required)"
)
def lm_eval_vllm_pipeline(
    model_path: str = "facebook/opt-125m",
    limit: int = 100,
    batch_size: int = 1,
    max_model_len: int = 2048,
    gpu_memory_utilization: float = 0.8,
    dtype: str = "auto",
    # Resource settings
    cpu_request: str = "4",
    cpu_limit: str = "8",
    memory_request: str = "16Gi",
    memory_limit: str = "32Gi",
    gpu_count: int = 1,
):
    eval_task = evaluate_model(
        model_path=model_path,
        limit=limit,
        batch_size=batch_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
    )
    
    # CPU/Memory
    eval_task.set_cpu_request(cpu_request)
    eval_task.set_cpu_limit(cpu_limit)
    eval_task.set_memory_request(memory_request)
    eval_task.set_memory_limit(memory_limit)
    
    # GPU
    eval_task.set_accelerator_type("nvidia.com/gpu")
    eval_task.set_accelerator_limit(gpu_count)


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        evaluate_model,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
    print(f"Compiled to: {__file__.replace('.py', '_component.yaml')}")

