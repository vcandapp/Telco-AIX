# Qwen3-Embedding-8B Deployment

Simple deployment for Qwen3-Embedding-8B model using Text Embeddings Inference (TEI).

## Files

- `qwen3-embedding-tei-deployment.yaml` - Complete deployment with ServiceAccount, SCC, ConfigMap, Deployment, Service, and Route

## Deploy

```bash
oc apply -f qwen3-embedding-tei-deployment.yaml
```

## API Access

**Endpoint:** `https://qwen3-embedding-tei-route-tme-aix.apps.acmhub.narlabs.io`

**Health Check:**
```bash
curl -k https://qwen3-embedding-tei-route-tme-aix.apps.acmhub.narlabs.io/health
```

**Generate Embeddings:**
```bash
curl -k -X POST https://qwen3-embedding-tei-route-tme-aix.apps.acmhub.narlabs.io/embed \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer tme-aix-embedding-2024-secure-token" \
  -d '{"inputs": ["Your text here"]}'
```

## Security

API requires authentication with Bearer token. Without token returns `401 Unauthorized`.

## Components

- **Secret:** `qwen3-embedding-api-token` - API authentication token
- **ServiceAccount:** `qwen3-embedding-tei-sa`
- **SCC:** anyuid ClusterRoleBinding 
- **ConfigMap:** `qwen3-embedding-tei-config`
- **Deployment:** `qwen3-embedding-tei`
- **Service:** `qwen3-embedding-tei-service`
- **Route:** `qwen3-embedding-tei-route`

## Model Details

- **Model:** Qwen/Qwen3-Embedding-8B
- **Engine:** Text Embeddings Inference (TEI)
- **GPU:** Any available NVIDIA GPU
- **Max Context:** 8,192 tokens
- **Embedding Dim:** 4,096

## GPU Memory Planning

### Memory Usage Formula

**Base Model Memory:**
```
Model Size = 8B parameters × 2 bytes (float16) = 16GB
```

**Context Processing Memory:**
```
Context Memory = Max Context × Embedding Dim × Batch Size × 4 bytes (float32)
Example: 8,192 × 4,096 × 32 × 4 = 4.3GB
```

**Total GPU Memory Required:**
```
Total = Model Size + Context Memory + Overhead (2-4GB)
Example: 16GB + 4.3GB + 3GB = ~23GB
```

### Scaling Parameters

**Increase Max Context (8,192 → 16,384):**
- Memory impact: +4.3GB per doubling
- Edit ConfigMap: `MAX_BATCH_TOKENS: "65536"`
- Requires: Additional 4-8GB GPU memory

**Increase Batch Size (32 → 64):**
- Memory impact: +4.3GB per doubling  
- Edit ConfigMap: `MAX_CONCURRENT_REQUESTS: "64"`
- Edit ConfigMap: `MAX_CLIENT_BATCH_SIZE: "128"`
- Requires: Additional 4-8GB GPU memory

### GPU Memory Requirements

| GPU Memory | Max Context | Batch Size | Use Case |
|------------|-------------|------------|----------|
| 16GB | 4,096 | 16 | Basic inference |
| 24GB | 8,192 | 32 | **Current setup** |
| 32GB | 16,384 | 32 | Long documents |
| 48GB | 16,384 | 64 | High throughput |
| 80GB | 32,768 | 64 | Enterprise scale |

### Configuration Updates

To modify parameters, edit the ConfigMap in deployment YAML:

```yaml
data:
  MAX_BATCH_TOKENS: "32768"        # Max Context × 4
  MAX_CONCURRENT_REQUESTS: "32"    # Batch size
  MAX_CLIENT_BATCH_SIZE: "64"      # Client batch limit
```

Then redeploy:
```bash
oc apply -f qwen3-embedding-tei-deployment.yaml
```