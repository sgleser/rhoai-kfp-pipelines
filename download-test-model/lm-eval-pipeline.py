#!/usr/bin/env python3
"""
Simple Kubeflow Pipeline for evaluating LLMs using lm-eval harness with Unitxt.
Connects to an existing InferenceService endpoint.
Uses German language tasks (XNLI-de, MLQA-de).
"""

from kfp import dsl
import kfp


@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
    packages_to_install=[
        "lm-eval>=0.4.0",
        "unitxt",
    ],
)
def evaluate_model(
    inference_url: str,
    model_name: str,
    output_metrics: dsl.Output[dsl.Metrics],
    output_results: dsl.Output[dsl.Artifact],
    limit: int = 100,
):
    """
    Evaluates an LLM using lm-eval harness with Unitxt tasks.
    
    Args:
        inference_url: URL of the InferenceService
        model_name: Model name in the InferenceService
        limit: Max examples per task
    """
    import os
    import json
    import logging

    from lm_eval.api.registry import get_model
    from lm_eval.evaluator import evaluate
    from lm_eval.tasks import get_task_dict
    from lm_eval.tasks.unitxt import task

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    print("=" * 60)
    print("LM-EVAL BENCHMARK (German - Unitxt)")
    print("=" * 60)
    print(f"URL: {inference_url}")
    print(f"Model: {model_name}")
    print(f"Limit: {limit}")
    print("=" * 60)

    # Define Unitxt tasks (German)
    UNITXT_TASKS = [
        # German NLI (Natural Language Inference) from XNLI dataset
        {
            "task": "xnli_german",
            "recipe": "card=cards.xnli.de,template=templates.classification.multi_class.relation.default",
            "group": "classification",
            "output_type": "generate_until",
        },
        # German reading comprehension from MLQA dataset
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

    # Connect to InferenceService
    base_url = inference_url.rstrip('/')
    model_args = {
        "model": model_name,
        "base_url": f"{base_url}/v1/completions",
        "num_concurrent": 1,
        "max_retries": 3,
        "tokenized_requests": False,
    }

    model_class = get_model("local-completions")
    loaded_model = model_class.create_from_arg_obj(model_args, {"batch_size": 1})
    logger.info("Connected to InferenceService")

    # Run evaluation
    logger.info("Running evaluation...")
    results = evaluate(
        lm=loaded_model,
        task_dict=task_dict,
        limit=limit,
    )

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
    name="lm-eval-german",
    description="Evaluate LLM using lm-eval with German Unitxt tasks (XNLI-de, MLQA-de)"
)
def lm_eval_pipeline(
    inference_url: str,
    model_name: str,
    limit: int = 100,
):
    eval_task = evaluate_model(
        inference_url=inference_url,
        model_name=model_name,
        limit=limit,
    )
    eval_task.set_cpu_request("2")
    eval_task.set_cpu_limit("4")
    eval_task.set_memory_request("4Gi")
    eval_task.set_memory_limit("8Gi")


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        evaluate_model,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
    print(f"Compiled to: {__file__.replace('.py', '_component.yaml')}")
