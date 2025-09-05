# Telco Intent Classification with Qwen3-4B-Instruct

A supervised fine-tuning (SFT) project for telecommunications customer intent classification using the Qwen3-4B-Instruct model. This solution classifies customer queries into predefined intents for improved customer service automation.

## Quick Use with RHOAI with vLLM KServe RunTime

OCI URL: `oci://docker.io/efatnar/modelcar-qwen3-4b-sft:latest`


## Fine Tuning
### Installation
```bash
pip install -r requirements.txt
```

Required packages:
- `training_hub` - Training framework for supervised fine-tuning
- `requests` - HTTP library for API testing
- `urllib3` - HTTP client for Python

## 🗂️ Project Structure

```
intclass/
├── data/
│   └── cic.jsonl                 # Training dataset (9,403 samples)
├── docker/
│   ├── artifactory.txt           # Docker build instructions
│   ├── extract_model.py          # Model extraction utility
│   └── models/
│       └── touch.txt             # Placeholder for model files
├── qa/
│   └── qa_qwen_intent_classifier.py  # Comprehensive QA test suite
├── sft_qwen3-4b-instruct.py     # Main training script
├── test_trained_model.py        # Local model testing
├── requirements.txt              # Python dependencies
```

## 📊 Dataset

The training dataset (`data/cic.jsonl`) contains 9,403 conversation samples in JSONL format. Each sample follows this structure:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "Classify the user question into one of the predefined intents. Respond with only the intent name."
    },
    {
      "role": "user",
      "content": "follow up on ticket ?"
    },
    {
      "role": "assistant",
      "content": "ComplaintStatusInquiry"
    }
  ]
}
```

### Supported Intent Categories
- `5GFAQ` - 5G network inquiries
- `ComplaintStatusInquiry` - Ticket/complaint status checks
- `RetrieveSmilesInformation` - Rewards program
- `Migration` - Account type changes
- `CurrentPlanActiveProdsSrvcsInquiry` - Contract and plan details
- `SimManagement` - SIM card operations
- `AvailableBalanceRequest` - Balance inquiries
- `RetrieveThirdPartyBilling` - Billing and charges
- And more...

## Fine-Tuning

### Basic Fine-Tuning
```bash
python sft_qwen3-4b-instruct.py \
    --data-path data/cic.jsonl \
    --ckpt-output-dir output/
```

### Advanced Configuration
```bash
python sft_qwen3-4b-instruct.py \
    --data-path data/cic.jsonl \
    --ckpt-output-dir output/ \
    --num-epochs 3 \
    --max-tokens-per-gpu 8192 \
    --effective-batch-size 16 \
    --learning-rate 2e-6 \
    --max-seq-len 8192
```

### Training Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model-path` | Qwen/Qwen3-4B-Instruct-2507 | Base model |
| `--num-epochs` | 3 | Training epochs |
| `--max-tokens-per-gpu` | 8192 | Max tokens per GPU |
| `--effective-batch-size` | 16 | Batch size for single GPU |
| `--learning-rate` | 2e-6 | Learning rate (conservative for stability) |
| `--max-seq-len` | 8192 | Maximum sequence length |

### GPU Optimizations
- **Flash Attention**: Disabled for Blackwell compatibility
- **Memory**: Conservative settings to avoid OOM
- **Single GPU**: Optimized for single RTX PRO 6000
- **Checkpointing**: Full state saved at each epoch

## 🧪 Testing

### Local Model Testing
Test the fine-tuned model locally:
```bash
python test_trained_model.py
```

This script:
- Loads the model from `output/hf_format/samples_*/`
- Tests intent classification on predefined queries
- Reports GPU memory usage

### API Testing
For deployed models served via vLLM:
```bash
python qa/qa_qwen_intent_classifier.py
```

## 🔧 Troubleshooting

### Storage Requirements 
**Important:** Each training epoch saves a full model checkpoint (~8-16GB per checkpoint). With 10 epochs, you'll need:
- **Minimum**: 100GB free disk space
- **Recommended**: 150-200GB for safe operation
- **Per epoch**: ~8-16GB depending on model size

Storage optimization strategies:
- Use `--save-samples 0` to disable sample-based checkpointing
- Reduce `--num-epochs` if storage is limited
- Clean up intermediate checkpoints after training
- Use `/dev/shm` for temporary data (RAM disk) as configured in the script

### Memory Issues
If encountering OOM errors:
- Reduce `--max-tokens-per-gpu` (e.g., 4096)
- Lower `--effective-batch-size` (e.g., 8)
- Decrease `--max-seq-len` (e.g., 4096)

### GPU Configuration
Verify GPU availability:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Training Failures
The training script provides detailed error messages and suggestions for:
- Memory optimization
- GPU configuration issues
- Data format problems
- Storage space exhaustion

