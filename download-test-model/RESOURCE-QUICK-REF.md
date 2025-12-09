# Resource Quick Reference Card

## Model Size → Resource Settings

### 🔹 Tiny Models (125M-350M params)
*Examples: facebook/opt-125m, gpt2*

```yaml
# Download Component
download_cpu_request: "1"
download_cpu_limit: "2"
download_memory_request: "2Gi"
download_memory_limit: "4Gi"

# vLLM Server Component
vllm_cpu_request: "2"
vllm_cpu_limit: "4"
vllm_memory_request: "4Gi"
vllm_memory_limit: "8Gi"
tensor_parallel_size: 0  # CPU-only, no GPU needed
```

**PVC Size**: 10Gi  
**Download Time**: 1-2 min  
**GPU**: Not required

---

### 🔸 Small Models (1B-3B params)
*Examples: facebook/opt-1.3b, facebook/opt-2.7b*

```yaml
# Download Component
download_cpu_request: "2"
download_cpu_limit: "4"
download_memory_request: "4Gi"
download_memory_limit: "8Gi"

# vLLM Server Component
vllm_cpu_request: "4"
vllm_cpu_limit: "8"
vllm_memory_request: "8Gi"
vllm_memory_limit: "12Gi"
tensor_parallel_size: 1  # 1 GPU recommended
```

**PVC Size**: 25Gi  
**Download Time**: 5-10 min  
**GPU**: Recommended (8GB+ VRAM)

---

### 🔶 Medium Models (7B params) ⭐ DEFAULT
*Examples: mistralai/Mistral-7B-v0.1, meta-llama/Llama-2-7b-hf*

```yaml
# Download Component
download_cpu_request: "2"
download_cpu_limit: "4"
download_memory_request: "4Gi"
download_memory_limit: "8Gi"

# vLLM Server Component
vllm_cpu_request: "4"
vllm_cpu_limit: "8"
vllm_memory_request: "8Gi"
vllm_memory_limit: "16Gi"
tensor_parallel_size: 1  # 1 GPU required
```

**PVC Size**: 50Gi  
**Download Time**: 10-20 min  
**GPU**: Required (24GB VRAM: A10, L4, A100-40GB)

---

### 🔴 Large Models (13B params)
*Examples: meta-llama/Llama-2-13b-hf*

```yaml
# Download Component
download_cpu_request: "2"
download_cpu_limit: "4"
download_memory_request: "8Gi"
download_memory_limit: "16Gi"

# vLLM Server Component
vllm_cpu_request: "6"
vllm_cpu_limit: "12"
vllm_memory_request: "16Gi"
vllm_memory_limit: "32Gi"
tensor_parallel_size: 2  # 2 GPUs required
```

**PVC Size**: 100Gi  
**Download Time**: 20-40 min  
**GPU**: 2x GPUs (24GB+ VRAM each: A10, A100)

---

### 🟣 Very Large Models (70B+ params)
*Examples: meta-llama/Llama-2-70b-hf*

```yaml
# Download Component
download_cpu_request: "4"
download_cpu_limit: "8"
download_memory_request: "16Gi"
download_memory_limit: "32Gi"

# vLLM Server Component
vllm_cpu_request: "12"
vllm_cpu_limit: "24"
vllm_memory_request: "32Gi"
vllm_memory_limit: "64Gi"
tensor_parallel_size: 4  # 4 GPUs required
```

**PVC Size**: 200Gi  
**Download Time**: 40+ min  
**GPU**: 4x GPUs (40GB VRAM: A100-40GB) or 2x A100-80GB

---

## GPU Quick Reference

| GPU Model | VRAM | Best For |
|-----------|------|----------|
| T4 | 16GB | Up to 7B (with quantization) |
| L4 | 24GB | 7B models |
| A10 / A10G | 24GB | 7B models |
| V100 | 32GB | Up to 13B |
| A100-40GB | 40GB | Up to 13B, or 70B with 4 GPUs |
| A100-80GB | 80GB | Up to 70B, or 2 GPUs for 70B |
| H100 | 80GB | Any size, best performance |

## Memory Optimization Tricks

### 🔧 Out of Memory? Try these:

1. **Reduce sequence length**:
   ```yaml
   max_model_len: 2048  # instead of 4096
   ```

2. **Use quantization** (in vLLM startup args):
   ```python
   # 8-bit: saves 50% memory
   "--quantization", "bitsandbytes"
   
   # 4-bit: saves 75% memory
   "--quantization", "awq"
   ```

3. **Reduce GPU memory utilization**:
   ```python
   "--gpu-memory-utilization", "0.85"  # default is 0.9
   ```

4. **Use multi-GPU**:
   ```yaml
   tensor_parallel_size: 2  # split across 2 GPUs
   ```

## Common Issues & Fixes

| Problem | Solution |
|---------|----------|
| Pod Pending | Reduce CPU/memory requests |
| OOMKilled | Increase memory limits |
| CUDA OOM | Use quantization or multi-GPU |
| Slow download | Check network, increase CPU |
| CPU throttling | Increase CPU limits |

## Copy-Paste Commands

### Check resource usage:
```bash
kubectl top pods
kubectl describe pod <pod-name> | grep -A 5 "Requests\|Limits"
```

### Check for OOM:
```bash
kubectl get events | grep OOM
kubectl describe pod <pod-name> | grep -i oom
```

### Check GPU availability:
```bash
kubectl describe nodes | grep -A 5 "nvidia.com/gpu"
```

### Port forward for testing:
```bash
kubectl port-forward <vllm-pod> 8000:8000
curl http://localhost:8000/health
```

---

## 📖 Full Documentation

- [RESOURCES.md](RESOURCES.md) - Complete resource guide
- [README.md](README.md) - Full documentation
- [QUICKSTART.md](QUICKSTART.md) - Get started quickly

---

**Last Updated**: December 2025  
**Pipeline Version**: 1.0





