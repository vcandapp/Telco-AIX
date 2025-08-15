# Embedding GenAI Model Deployment with Text Embeddings Inference (TEI).

Simple deployment for Qwen3-Embedding-8B model using TEI.

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
