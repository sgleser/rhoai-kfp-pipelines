# Quick Start Guide

Get up and running in 5 minutes!

## Prerequisites

- Kubernetes cluster with Kubeflow Pipelines
- Storage class supporting ReadWriteMany (RWX)
- GPU node (optional, but recommended for models > 1B params)

## Setup (5 minutes)

### 1. Create Storage

```bash
# Edit storage class in pvc-storage.yaml
kubectl apply -f pvc-storage.yaml
```

### 2. Deploy Pipeline

```bash
# Compile
python download-test.py

# Upload model_test_pipeline.yaml to Kubeflow UI
# Or use CLI:
kfp pipeline upload -p model_test_pipeline.yaml
```

### 3. Run Pipeline

In Kubeflow UI, create run with defaults or:

```bash
kfp run submit \
  -e your-experiment \
  -p model_name=facebook/opt-125m \
  -p pvc_name=model-storage-pvc \
  model_test_pipeline.yaml
```

### 4. Test with Promptfoo

```bash
# Port forward to vLLM server
kubectl port-forward <vllm-pod> 8000:8000 &

# Run tests
./run-promptfoo-test.sh

# View results
cd promptfoo-results && promptfoo view
```

## Common Commands

```bash
# Check PVC status
kubectl get pvc

# Watch pipeline pods
kubectl get pods -w

# View logs
kubectl logs -f <pod-name>

# Port forward to server
kubectl port-forward <pod-name> 8000:8000

# Test server health
curl http://localhost:8000/health

# Run Promptfoo tests
VLLM_URL=http://localhost:8000 ./run-promptfoo-test.sh
```

## Test Different Models

Small (< 1 min download):
- `facebook/opt-125m`
- `gpt2`

Medium (2-5 min):
- `facebook/opt-1.3b`
- `facebook/opt-2.7b`

Large (10-20 min, needs GPU):
- `mistralai/Mistral-7B-v0.1`
- `mistralai/Mistral-7B-Instruct-v0.2`

## File Structure

```
.
├── download-test.py          # Pipeline definition
├── Dockerfile                # Container image
├── pvc-storage.yaml          # Storage config
├── promptfoo-config.json     # Test suite
├── run-promptfoo-test.sh     # Test runner
├── README.md                 # Full docs
├── example-usage.md          # Detailed examples
└── QUICKSTART.md            # This file
```

## Troubleshooting

**PVC won't bind?**
- Check storage class: `kubectl get sc`
- Ensure it supports ReadWriteMany

**Download fails?**
- Check internet access from pods
- For private models, add HF token

**vLLM won't start?**
- Check GPU availability: `kubectl describe nodes`
- Reduce `max_model_len` parameter
- Check logs: `kubectl logs <pod>`

**Promptfoo tests fail?**
- Ensure port-forward is active
- Check vLLM health: `curl http://localhost:8000/health`
- Increase timeout in config

## Quick Architecture

```
┌─────────────────────────────────────────┐
│         Kubeflow Pipeline               │
│                                         │
│  ┌──────────┐      ┌──────────────┐   │
│  │ Download │─────▶│ vLLM Server  │   │
│  │  Model   │      │   (GPU)      │   │
│  └─────┬────┘      └──────┬───────┘   │
│        │                  │            │
└────────┼──────────────────┼────────────┘
         │                  │
         └─────┬────────────┘
               │
      ┌────────▼─────────┐
      │   PVC Storage    │
      │  (ReadWriteMany) │
      └──────────────────┘
               ▲
               │
      ┌────────┴─────────┐
      │    Promptfoo     │
      │   Test Results   │
      └──────────────────┘
```

## Next Steps

1. ✅ Run default pipeline with `facebook/opt-125m`
2. ✅ Test with Promptfoo
3. ✅ Try a larger model
4. ✅ Customize test suite
5. ✅ Read full docs in README.md
6. ✅ Check examples in example-usage.md

## Resources

- [Full Documentation](README.md)
- [Usage Examples](example-usage.md)
- [vLLM Docs](https://docs.vllm.ai/)
- [Promptfoo Docs](https://promptfoo.dev/)
- [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/)





