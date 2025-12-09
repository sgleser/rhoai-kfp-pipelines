# vLLM Model Testing Pipeline

A Kubeflow Pipeline for downloading models from HuggingFace and starting a vLLM server for testing. Uses PersistentVolumeClaim (PVC) for shared storage between pipeline components. Includes automated testing with Promptfoo.

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes
- **[example-usage.md](example-usage.md)** - Detailed usage examples and scenarios
- **[RESOURCES.md](RESOURCES.md)** - CPU, memory, and GPU resource requirements guide
- **[README.md](README.md)** - This file (complete reference)

## Table of Contents

- [Architecture](#architecture)
- [Files](#files)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Pipeline Parameters](#pipeline-parameters)
- [Model Examples](#model-examples)
- [Storage Requirements](#storage-requirements)
- [Resource Requirements](#resource-requirements)
- [GPU Configuration](#gpu-configuration)
- [Testing the vLLM Server](#testing-the-vllm-server)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)
- [OpenShift AI / RHOAI Integration](#openshift-ai--rhoai-integration)
- [Next Steps](#next-steps)

## Architecture

```
┌─────────────────┐      ┌──────────────────┐
│  Download Model │      │  vLLM Server     │
│  Component      │─────▶│  Component       │
└────────┬────────┘      └────────┬─────────┘
         │                        │
         └────────┬───────────────┘
                  │
         ┌────────▼────────┐
         │   PVC Storage   │
         │  (Shared Volume)│
         └─────────────────┘
```

## Files

### Core Pipeline Files
- `download-test.py` - Main Kubeflow pipeline definition
- `Dockerfile` - Container image with vLLM and HuggingFace tools
- `pvc-storage.yaml` - PersistentVolumeClaim for model storage
- `build.sh` - Script to build and tag the container image

### Testing Files
- `promptfoo-config.json` - Promptfoo test configuration
- `run-promptfoo-test.sh` - Standalone script to run Promptfoo tests

### Documentation Files
- `README.md` - Complete documentation (this file)
- `QUICKSTART.md` - 5-minute quick start guide
- `example-usage.md` - Detailed usage examples
- `RESOURCES.md` - Comprehensive resource requirements guide
- `RESOURCE-QUICK-REF.md` - Quick reference card for resource settings

## Prerequisites

1. **Kubernetes Cluster** with Kubeflow Pipelines installed
2. **Storage Class** that supports ReadWriteMany (RWX) access mode
   - OpenShift Data Foundation (ODF/OCS)
   - NFS
   - CephFS
   - Or any RWX-capable storage
3. **GPU nodes** (recommended for larger models)
4. **Container Registry** access (Quay.io, Docker Hub, etc.)

## Setup

### 1. Create the PVC

First, update the storage class in `pvc-storage.yaml` to match your cluster:

```bash
# Check available storage classes
kubectl get storageclass

# Edit pvc-storage.yaml and update:
# - namespace (if not using 'default')
# - storage size (based on your model requirements)
# - storageClassName (to match your cluster)

# Create the PVC
kubectl apply -f pvc-storage.yaml

# Verify PVC is created and bound
kubectl get pvc model-storage-pvc
```

### 2. Build and Push Custom Image (Optional)

If you want to use a custom base image with additional tools:

```bash
# Edit build.sh and update registry/namespace
vim build.sh

# Build the image
./build.sh

# Push to registry
docker push <your-registry>/vllm-model-test:latest

# Update CONTAINER_IMAGE in download-test.py
vim download-test.py
# Change: CONTAINER_IMAGE = "<your-registry>/vllm-model-test:latest"
```

Alternatively, use the default `vllm/vllm-openai:latest` image (it already has most dependencies).

### 3. Compile the Pipeline

```bash
# Install KFP SDK if not already installed
pip install kfp

# Compile the pipeline
python download-test.py

# This creates: model_test_pipeline.yaml
```

### 4. Upload to Kubeflow

1. Open your Kubeflow Pipelines UI
2. Go to **Pipelines** → **Upload Pipeline**
3. Upload `model_test_pipeline.yaml`
4. Create a new run with your desired parameters

## Pipeline Parameters

### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `facebook/opt-125m` | HuggingFace model ID to download |
| `pvc_name` | `model-storage-pvc` | Name of the PVC for storage |
| `cache_dir` | `/mnt/models` | Mount path for the PVC |
| `server_port` | `8000` | Port for vLLM OpenAI API server |
| `tensor_parallel_size` | `1` | Number of GPUs to use |
| `max_model_len` | `2048` | Maximum sequence length |
| `run_promptfoo` | `false` | Enable Promptfoo testing |
| `test_prompt` | `Once upon a time` | Test prompt (for basic test) |

### Resource Parameters

#### Download Component Resources

| Parameter | Default | Description |
|-----------|---------|-------------|
| `download_cpu_request` | `"2"` | CPU cores requested for download |
| `download_cpu_limit` | `"4"` | Maximum CPU cores for download |
| `download_memory_request` | `"4Gi"` | Memory requested for download |
| `download_memory_limit` | `"8Gi"` | Maximum memory for download |

#### vLLM Server Component Resources

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vllm_cpu_request` | `"4"` | CPU cores requested for vLLM server |
| `vllm_cpu_limit` | `"8"` | Maximum CPU cores for vLLM server |
| `vllm_memory_request` | `"8Gi"` | Memory requested for vLLM server |
| `vllm_memory_limit` | `"16Gi"` | Maximum memory for vLLM server |

**Note**: GPU allocation is controlled by `tensor_parallel_size`. See [RESOURCES.md](RESOURCES.md) for detailed recommendations based on model size.

## Model Examples

### Small Models (Good for Testing)
- `facebook/opt-125m` - 125M parameters, ~500MB
- `facebook/opt-350m` - 350M parameters, ~1.5GB
- `gpt2` - 124M parameters, ~500MB

### Medium Models
- `facebook/opt-1.3b` - 1.3B parameters, ~5GB
- `facebook/opt-2.7b` - 2.7B parameters, ~10GB

### Large Models (Requires GPU)
- `meta-llama/Llama-2-7b-hf` - 7B parameters, ~15GB (requires HF token)
- `mistralai/Mistral-7B-v0.1` - 7B parameters, ~15GB
- `mistralai/Mistral-7B-Instruct-v0.2` - 7B parameters, ~15GB

## Storage Requirements

| Model Size | Approximate Storage | Recommended PVC Size |
|------------|-------------------|---------------------|
| 125M - 350M | 0.5 - 2 GB | 10 GB |
| 1B - 3B | 5 - 12 GB | 25 GB |
| 7B | 13 - 18 GB | 50 GB |
| 13B | 25 - 30 GB | 100 GB |
| 70B | 130 - 150 GB | 200 GB |

## Resource Requirements

**Default resource allocations are suitable for 7B parameter models.**

### Quick Reference

| Component | CPU Request | CPU Limit | Memory Request | Memory Limit | GPU |
|-----------|-------------|-----------|----------------|--------------|-----|
| Download | 2 cores | 4 cores | 4Gi | 8Gi | No |
| vLLM Server | 4 cores | 8 cores | 8Gi | 16Gi | 1 (optional) |
| Promptfoo | 1 core | 2 cores | 2Gi | 4Gi | No |

### Adjusting for Different Model Sizes

- **Small models (< 1B params)**: Use 50% of default resources
- **Medium models (1-7B params)**: Use default resources
- **Large models (7-30B params)**: Double memory and use multi-GPU
- **Very large models (70B+ params)**: See [RESOURCES.md](RESOURCES.md) for detailed guide

**📖 For comprehensive resource planning, see [RESOURCES.md](RESOURCES.md)**

This guide includes:
- Detailed tables for all model sizes
- GPU memory requirements
- Multi-GPU configurations
- Memory optimization techniques
- Troubleshooting resource issues

## GPU Configuration

For models that require GPUs, the pipeline automatically sets GPU limits based on `tensor_parallel_size`:

```python
# In the pipeline, GPU is set automatically
kubernetes.set_gpu_limit(vllm_task, tensor_parallel_size)
```

For multi-GPU setups, increase `tensor_parallel_size`:
- `tensor_parallel_size=1` - Single GPU
- `tensor_parallel_size=2` - 2 GPUs
- `tensor_parallel_size=4` - 4 GPUs (for very large models)

## Testing the vLLM Server

Once the pipeline starts the vLLM server, you can test it in multiple ways:

### 1. Port Forward to the Server

```bash
# Find the pod name
kubectl get pods | grep vllm

# Port forward
kubectl port-forward <pod-name> 8000:8000
```

### 2. Quick Test with curl

```bash
# Health check
curl http://localhost:8000/health

# Test completion
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "prompt": "Once upon a time",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

### 3. Test with Python

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/completions",
    json={
        "model": "facebook/opt-125m",
        "prompt": "Once upon a time",
        "max_tokens": 50
    }
)

print(response.json())
```

### 4. Automated Testing with Promptfoo

**Promptfoo** is a comprehensive LLM testing and evaluation framework that can run automated test suites against your model.

#### Install Promptfoo

```bash
npm install -g promptfoo
```

#### Run Tests

```bash
# Make sure vLLM server is accessible (port-forward if needed)
kubectl port-forward <pod-name> 8000:8000

# Run the test script
./run-promptfoo-test.sh

# Or manually
promptfoo eval -c promptfoo-config.json
```

#### View Results in Browser

```bash
promptfoo view
```

This opens an interactive web UI showing:
- Test pass/fail status
- Model outputs for each test case
- Performance metrics
- Side-by-side comparisons (if testing multiple models)

#### Customize Tests

Edit `promptfoo-config.json` to add your own test cases:

```json
{
  "tests": [
    {
      "description": "Your custom test",
      "vars": {
        "topic": "your topic here"
      },
      "assert": [
        {
          "type": "contains",
          "value": "expected text"
        },
        {
          "type": "llm-rubric",
          "value": "Response should be accurate and helpful"
        }
      ]
    }
  ]
}
```

#### Promptfoo Test Types

- **contains**: Check if output contains specific text
- **contains-any**: Check if output contains any of the given options
- **javascript**: Custom JavaScript validation
- **llm-rubric**: Use an LLM to evaluate output quality
- **similar**: Semantic similarity check
- **cost**: Token/cost assertions
- **latency**: Response time checks

#### Benefits of Promptfoo Testing

✅ **Automated**: Run comprehensive test suites automatically  
✅ **Regression Detection**: Catch quality degradation between models  
✅ **Comparison**: Test multiple models side-by-side  
✅ **CI/CD Integration**: Integrate into your deployment pipeline  
✅ **Rich Assertions**: Multiple validation methods  
✅ **Results Tracking**: Historical test results and analytics

## Troubleshooting

### PVC Not Binding
- Check if your cluster has a storage class that supports ReadWriteMany (RWX)
- Verify storage class exists: `kubectl get storageclass`
- Check PVC status: `kubectl describe pvc model-storage-pvc`

### Download Fails
- Check internet connectivity from the pod
- For private models, you may need to add HuggingFace token authentication
- Increase memory limits if OOM errors occur

### vLLM Server Won't Start
- Check GPU availability: `kubectl describe node | grep -A 5 "Allocated resources"`
- Verify model was downloaded: `kubectl exec <pod> -- ls /mnt/models`
- Check logs: `kubectl logs <pod-name>`
- Reduce `max_model_len` if running out of GPU memory

### Out of Memory
- Increase PVC size in `pvc-storage.yaml`
- Increase memory limits in pipeline definition
- Use a smaller model for testing
- Enable CPU offloading in vLLM (advanced)

## Advanced Configuration

### Running Promptfoo in the Pipeline

The pipeline includes a Promptfoo testing component that can be enabled. To use it:

1. Edit `download-test.py` and uncomment the Promptfoo section in the pipeline:

```python
# Uncomment this block in the pipeline definition:
with dsl.Condition(run_promptfoo == True):
    promptfoo_task = test_with_promptfoo(
        server_url=f"http://vllm-service:{server_port}",
        model_name=download_task.output,
        config_dir=f"{cache_dir}/promptfoo"
    )
    # ... rest of the configuration
```

2. When running the pipeline, set `run_promptfoo` parameter to `True`

3. Results will be saved to the PVC at `/mnt/models/promptfoo/`

**Note**: For in-pipeline testing to work, you'll need to expose the vLLM server as a Kubernetes Service so the Promptfoo component can reach it.

### Creating a Kubernetes Service for vLLM

To enable communication between pipeline components:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
spec:
  selector:
    # Add labels to match your vLLM pod
    component: vllm-server
  ports:
    - protocol: TCP
      port: 8000
      targetPort: 8000
```

### Adding HuggingFace Authentication

For private models (like Llama-2), add a secret:

```bash
# Create secret with HF token
kubectl create secret generic huggingface-token \
  --from-literal=token=<your-hf-token>

# Update download_model component to use the secret
```

Then in the pipeline, mount the secret:

```python
kubernetes.use_secret_as_env(
    download_task,
    secret_name='huggingface-token',
    secret_key_to_env={'token': 'HF_TOKEN'}
)
```

### Custom vLLM Arguments

Modify the `start_vllm_server` component to add custom vLLM arguments:

```python
cmd = [
    "python", "-m", "vllm.entrypoints.openai.api_server",
    "--model", model_name,
    "--trust-remote-code",  # For custom models
    "--quantization", "awq",  # For quantized models
    "--dtype", "float16",  # Precision
    "--gpu-memory-utilization", "0.9",  # GPU memory usage
    # ... other arguments
]
```

### Custom Promptfoo Test Suites

Create domain-specific test suites for your use case:

**Code Generation Tests**:
```json
{
  "tests": [
    {
      "vars": {"task": "fibonacci sequence", "language": "Python"},
      "assert": [
        {"type": "contains", "value": "def"},
        {"type": "is-valid-openai-function-call"},
        {"type": "llm-rubric", "value": "Code should be correct and efficient"}
      ]
    }
  ]
}
```

**Safety/Toxicity Tests**:
```json
{
  "tests": [
    {
      "vars": {"prompt": "potentially sensitive topic"},
      "assert": [
        {"type": "not-contains", "value": ["offensive", "harmful"]},
        {"type": "moderation", "threshold": 0.5}
      ]
    }
  ]
}
```

**Performance Tests**:
```json
{
  "tests": [
    {
      "assert": [
        {"type": "latency", "threshold": 2000},
        {"type": "cost", "threshold": 0.01}
      ]
    }
  ]
}
```

## OpenShift AI / RHOAI Integration

This pipeline works seamlessly with Red Hat OpenShift AI (RHOAI):

1. Deploy to RHOAI Data Science Pipelines
2. Use ODF (OpenShift Data Foundation) for storage
3. Access GPU nodes via Node Feature Discovery
4. Integrate with RHOAI model serving after testing

## Next Steps

- Add automated testing component to the pipeline
- Integrate with MLflow for model tracking
- Create a separate pipeline for model serving deployment
- Add model evaluation metrics
- Implement A/B testing between models

## License

This pipeline configuration is provided as-is for testing purposes.

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/)
- [HuggingFace Hub](https://huggingface.co/docs/hub/index)
- [OpenShift AI](https://www.redhat.com/en/technologies/cloud-computing/openshift/openshift-ai)

