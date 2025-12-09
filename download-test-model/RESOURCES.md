# Resource Requirements Guide

This document provides recommendations for CPU and memory resource requests and limits for different model sizes.

## Understanding Kubernetes Resources

### Requests vs Limits

- **Request**: The amount of resources guaranteed for the container
  - Used by Kubernetes scheduler to decide where to place the pod
  - Container will always get at least this amount
  
- **Limit**: The maximum amount of resources the container can use
  - If exceeded, container may be throttled (CPU) or killed (memory)
  - Should be higher than request to allow bursts

### Best Practices

1. **Always set both requests and limits** to avoid resource contention
2. **Requests should be realistic** based on actual usage
3. **Limits should allow for peaks** but prevent runaway processes
4. **Use proper units**:
   - CPU: Cores as integers ("1", "2", "4") or millicores ("500m", "1000m")
   - Memory: Use "Mi" (mebibytes) or "Gi" (gibibytes), e.g., "4Gi", "8Gi"

## Recommended Resource Settings

### Download Component

The download component pulls models from HuggingFace. Resource needs scale with model size.

| Model Size | CPU Request | CPU Limit | Memory Request | Memory Limit | Download Time* |
|------------|-------------|-----------|----------------|--------------|----------------|
| < 500MB (opt-125m, gpt2) | 1 | 2 | 2Gi | 4Gi | 1-2 min |
| 500MB - 2GB (opt-350m) | 2 | 4 | 4Gi | 8Gi | 2-5 min |
| 2GB - 5GB (opt-1.3b) | 2 | 4 | 6Gi | 12Gi | 5-10 min |
| 5GB - 15GB (7B models) | 2 | 4 | 8Gi | 16Gi | 10-20 min |
| 15GB - 30GB (13B models) | 2 | 4 | 12Gi | 24Gi | 20-40 min |
| > 30GB (70B+ models) | 4 | 8 | 16Gi | 32Gi | 40+ min |

*Download times are approximate and depend on network speed

**Default Settings** (suitable for 7B models):
```yaml
download_cpu_request: "2"
download_cpu_limit: "4"
download_memory_request: "4Gi"
download_memory_limit: "8Gi"
```

### vLLM Server Component

The vLLM server loads the model into memory and serves inference requests. This requires significantly more resources.

| Model Size | CPU Request | CPU Limit | Memory Request | Memory Limit | GPU Required | VRAM Needed |
|------------|-------------|-----------|----------------|--------------|--------------|-------------|
| < 500MB | 2 | 4 | 4Gi | 8Gi | No (CPU works) | N/A |
| 500MB - 2GB | 2 | 4 | 6Gi | 12Gi | Optional | 2-4GB |
| 2GB - 5GB | 4 | 8 | 8Gi | 16Gi | Recommended | 4-8GB |
| 5GB - 15GB (7B) | 4 | 8 | 12Gi | 24Gi | **Required** | 12-20GB |
| 15GB - 30GB (13B) | 6 | 12 | 16Gi | 32Gi | **Required** | 20-30GB |
| 30GB - 50GB (30B) | 8 | 16 | 24Gi | 48Gi | **Required** (Multi-GPU) | 30-50GB |
| > 50GB (70B+) | 12 | 24 | 32Gi | 64Gi | **Required** (Multi-GPU) | 80GB+ |

**Default Settings** (suitable for 7B models with GPU):
```yaml
vllm_cpu_request: "4"
vllm_cpu_limit: "8"
vllm_memory_request: "8Gi"
vllm_memory_limit: "16Gi"
tensor_parallel_size: 1  # Number of GPUs
```

### GPU Considerations

#### GPU Memory Requirements

vLLM loads the entire model into GPU memory. Required VRAM depends on:
- **Model size** (parameters × precision)
- **Precision**: FP16 (~2 bytes/param), FP32 (~4 bytes/param)
- **KV cache**: Additional memory for context
- **Overhead**: ~10-20% for operations

**Example for 7B model (FP16)**:
- Model weights: 7B × 2 bytes = 14GB
- KV cache (4096 context): ~2-3GB
- Overhead: ~2GB
- **Total: ~18-20GB VRAM needed**

#### GPU Types

| GPU Model | VRAM | Suitable For | Notes |
|-----------|------|--------------|-------|
| T4 | 16GB | Up to 7B (tight) | Use with quantization |
| L4 | 24GB | Up to 7B | Good for 7B models |
| A10 | 24GB | Up to 7B | Good for 7B models |
| A10G | 24GB | Up to 7B | Good for 7B models |
| V100 | 16GB / 32GB | Up to 7B / 13B | Depends on variant |
| A100 | 40GB / 80GB | Up to 70B | 80GB for largest models |
| H100 | 80GB | Any size | Best performance |

#### Multi-GPU Setup

For large models, use tensor parallelism:

```yaml
# For 13B model on 2x A10 (24GB each)
tensor_parallel_size: 2
vllm_memory_request: "24Gi"
vllm_memory_limit: "48Gi"

# For 70B model on 4x A100 (40GB each)
tensor_parallel_size: 4
vllm_memory_request: "48Gi"
vllm_memory_limit: "96Gi"
```

### Promptfoo Testing Component

Much lighter resource requirements since it only sends API requests.

**Default Settings**:
```yaml
cpu_request: "1"
cpu_limit: "2"
memory_request: "2Gi"
memory_limit: "4Gi"
```

## Example Configurations

### Configuration 1: Small Model Testing (opt-125m)

```python
python download-test.py
# Then run with:
model_name: facebook/opt-125m
download_cpu_request: "1"
download_cpu_limit: "2"
download_memory_request: "2Gi"
download_memory_limit: "4Gi"
vllm_cpu_request: "2"
vllm_cpu_limit: "4"
vllm_memory_request: "4Gi"
vllm_memory_limit: "8Gi"
tensor_parallel_size: 0  # CPU-only
```

### Configuration 2: Medium Model (Mistral-7B)

```python
model_name: mistralai/Mistral-7B-Instruct-v0.2
download_cpu_request: "2"
download_cpu_limit: "4"
download_memory_request: "8Gi"
download_memory_limit: "16Gi"
vllm_cpu_request: "4"
vllm_cpu_limit: "8"
vllm_memory_request: "12Gi"
vllm_memory_limit: "24Gi"
tensor_parallel_size: 1  # Single GPU (24GB+ VRAM)
```

### Configuration 3: Large Model (Llama-2-13B)

```python
model_name: meta-llama/Llama-2-13b-chat-hf
download_cpu_request: "2"
download_cpu_limit: "4"
download_memory_request: "12Gi"
download_memory_limit: "24Gi"
vllm_cpu_request: "6"
vllm_cpu_limit: "12"
vllm_memory_request: "16Gi"
vllm_memory_limit: "32Gi"
tensor_parallel_size: 2  # 2 GPUs (24GB+ VRAM each)
```

### Configuration 4: Very Large Model (Llama-2-70B)

```python
model_name: meta-llama/Llama-2-70b-chat-hf
download_cpu_request: "4"
download_cpu_limit: "8"
download_memory_request: "16Gi"
download_memory_limit: "32Gi"
vllm_cpu_request: "12"
vllm_cpu_limit: "24"
vllm_memory_request: "32Gi"
vllm_memory_limit: "64Gi"
tensor_parallel_size: 4  # 4 GPUs (40GB VRAM each) or 2x 80GB
```

## Optimizing Resources

### Memory Optimization Techniques

1. **Quantization**: Reduce precision to save memory
   ```python
   # 8-bit quantization (2x memory savings)
   quantization: "bitsandbytes"
   
   # 4-bit quantization (4x memory savings)
   quantization: "awq"
   ```

2. **Reduce Max Sequence Length**:
   ```python
   max_model_len: 2048  # Instead of 4096
   ```

3. **GPU Memory Utilization**:
   ```python
   gpu_memory_utilization: 0.9  # Use 90% of GPU memory
   ```

### CPU Optimization

1. **Reduce Download Parallelism**: Lower CPU if network is bottleneck
2. **Increase for Large Models**: More CPU for faster tokenization
3. **Watch for Throttling**: Check if hitting CPU limits

### Cost Optimization

1. **Right-size resources**: Don't over-allocate
2. **Use spot/preemptible instances** for testing
3. **Cache models on PVC**: Avoid repeated downloads
4. **Schedule off-peak**: If testing in production cluster

## Monitoring Resource Usage

### Check Current Usage

```bash
# Watch pod resources
kubectl top pods

# Get detailed resource info
kubectl describe pod <pod-name> | grep -A 5 "Requests\|Limits"

# Check if pods are being OOMKilled
kubectl get events | grep OOM

# Check CPU throttling
kubectl describe pod <pod-name> | grep -i throttl
```

### Signs You Need More Resources

**Memory**:
- Pod gets OOMKilled
- `kubectl describe pod` shows OOM events
- Model fails to load with out-of-memory errors

**CPU**:
- Very slow download/loading times
- High CPU throttling percentage
- Requests timing out

**GPU**:
- CUDA out-of-memory errors
- Model fails to load
- Very slow inference

## Resource Quotas

If running in a namespace with resource quotas, ensure your quota can accommodate these resources:

```bash
# Check current quota
kubectl describe resourcequota -n <namespace>

# Typical quota needed for 7B model testing:
# - CPU: ~12 cores total
# - Memory: ~24Gi total
# - GPU: 1-2 devices
```

## Troubleshooting

### Pod Stuck in Pending

```bash
kubectl describe pod <pod-name>
# Look for: "Insufficient cpu" or "Insufficient memory"
```

**Solution**: Reduce resource requests or add more cluster capacity

### OOMKilled

```bash
kubectl describe pod <pod-name> | grep -A 5 "Last State"
# Shows: "OOMKilled"
```

**Solution**: Increase memory limits or use smaller model/quantization

### CPU Throttling

```bash
kubectl describe pod <pod-name>
# Check throttled percentage in metrics
```

**Solution**: Increase CPU limits if affecting performance

### GPU Out of Memory

```
CUDA out of memory error in logs
```

**Solution**: 
- Use quantization (4-bit/8-bit)
- Reduce `max_model_len`
- Use multi-GPU with `tensor_parallel_size`
- Choose smaller model

## References

- [Kubernetes Resource Management](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/)
- [vLLM Memory Requirements](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Model Memory Calculator](https://huggingface.co/docs/transformers/model_memory_anatomy)





