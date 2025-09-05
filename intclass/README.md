# Telco AI Intent Classification: A GenAI Transformation Journey

## Executive Summary

This project represents a comprehensive transformation of a telecommunications company's customer support system, replacing a traditional NLP-based intent classifier with a state-of-the-art Generative AI solution. Through systematic fine-tuning, advanced prompt engineering, and deployment on Red Hat OpenShift AI (RHOAI), we achieved **94.8% accuracy** in intent classification, surpassing the legacy system's performance while providing superior scalability and multilingual capabilities.

## Project Overview

### Business Challenge
- **Legacy System Limitations**: Traditional NLP classifiers struggled with:
  - Complex customer queries requiring contextual understanding
  - Multilingual support (English and Arabic)
  - Scalability constraints
  - High maintenance overhead for rule-based systems

### Solution Architecture
- **Base Model**: Qwen3-4B-Instruct (fine-tuned)
- **Infrastructure**: Red Hat OpenShift AI (RHOAI)
- **Serving Runtime**: vLLM with KServe
- **Training Framework**: Supervised Fine-Tuning (SFT) with training_hub

## Technical Implementation

### 1. Model Selection & Evaluation Journey

Our systematic evaluation process tested multiple models to identify the optimal solution:

| Model | Parameters | Accuracy | Infrastructure | Notes |
|-------|------------|----------|----------------|-------|
| Granite-7B-Lab | 7B | 19.96% | RHOAI | Baseline test |
| Granite-8B-Lab-v1 | 8B | 41.46% | RHOAI | With context definitions |
| Llama-3.2-3B | 3B | 49.05% | RHOAI | Lightweight option |
| Phi-4 | - | 78.62% | RHOAI | Microsoft model |
| Llama-4-Scout-17B | 17B | 83.32% | 4x43GB VRAM | Best pre-trained performance |
| Qwen3-32B (Full) | 32B | 87.10% | RHOAI | Large model baseline |
| **Qwen3-4B (Fine-tuned)** | **4B** | **94.8%** | **RHOAI** | **Our Final Stop** |

### 2. Fine-Tuning Process

#### Dataset Preparation
- **Training Samples**: 9,445 conversation examples
- **Format**: JSONL with structured conversation format
- **Languages**: English (primary) and Arabic support
- **Intent Categories**: 70+ predefined intents covering all customer service scenarios

#### Training Configuration
```python
# Optimized hyperparameters for RTX PRO 6000
{
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "epochs": 3,
    "learning_rate": 2e-6,
    "batch_size": 16,
    "max_seq_length": 8192,
    "temperature": 0.1,
    "flash_attention": false  # Disabled for Blackwell compatibility
}
```

#### Performance Optimization
- **Memory Management**: Conservative token limits to prevent OOM
- **Storage Efficiency**: Checkpoint management (100GB minimum required)
- **GPU Optimization**: Single RTX PRO 6000 configuration
- **Training Time**: ~3-4 hours per epoch

### 3. Prompt Engineering Strategy

Our comprehensive prompt engineering approach incorporated multiple optimization techniques:

#### Structured Intent Classification Prompt
```
You are an expert in understanding user intent. Your task is to classify the given user 
question into one of the following predefined intents. Respond with only the name of 
the intent and nothing else.

Important guidelines for classification:
1. Focus on the main topic, not peripheral mentions
2. Look for specific keywords that indicate the category
3. Choose the most specific applicable category
4. Always choose from the exact document library names listed
5. If unsure, use "AgentHandover" with confidence "low"
```

#### Key Improvements
1. **Context Definitions**: Added detailed intent descriptions (+25% accuracy)
2. **Rule-Based Guidelines**: Explicit classification rules for ambiguous cases
3. **Temperature Tuning**: Optimal at 0.1 for consistent responses
4. **Few-Shot Examples**: Included representative examples for each intent

### 4. Intent Categories

The system classifies customer queries into 70+ intents, including:

**Customer Service**
- `ComplaintStatusInquiry`: Ticket and complaint tracking
- `AvailableBalanceRequest`: Balance and usage inquiries
- `RetrieveCustomerBill`: Bill retrieval requests
- `BillComplaint`: Billing disputes and clarifications

**Technical Support**
- `ResolveTechnicalIssue`: Network and service troubleshooting
- `CreateReplaceRepairTicket`: Hardware issues
- `5GFAQ`: 5G network inquiries
- `SimManagement`: SIM card operations

**Account Management**
- `Migration`: Plan changes (prepaid/postpaid)
- `CustomerAccountDtlsPrefsUpdateRequest`: Profile updates
- `IDUpdateRequest`: Customer ID updates
- `AccountSuspensionRequest`: Temporary service suspension

**Value-Added Services**
- `RetrieveRewardsInformation`: Rewards program
- `RequestSubscriptionOrder`: Add-on subscriptions
- `PreferredNumber`: Premium number selection
- `AutopayRegistrationFAQ`: Payment automation

## Results & Performance

### Accuracy Improvements
```
Legacy NLP System:        ~75% accuracy
Initial LLM Tests:        38-83% (various models)
Fine-tuned Qwen3-4B:      94.8% accuracy (English)
                          81.44% accuracy (Arabic)
```

### Key Performance Metrics
- **Response Time**: <500ms average inference
- **Throughput**: 1000+ requests/minute
- **Model Size**: 4B parameters (optimized for edge deployment)
- **Memory Usage**: ~8-16GB VRAM
- **Multilingual Support**: English (primary), Arabic (secondary)

### Evaluation Results by Configuration

| Configuration | Test Set Size | Accuracy | Temperature | Notes |
|--------------|---------------|----------|-------------|-------|
| Baseline | 1000 | 83.12% | 0.5 | No fine-tuning |
| Improved Prompt | 1000 | 82.72% | 0.5 | Enhanced definitions |
| Optimal Temperature | 1000 | 83.32% | 0.9 | Temperature tuning |
| Fine-tuned Epoch 2 | 9445 | 87.09% | 0.1 | Early stopping point |
| Fine-tuned Epoch 3 | 9445 | 94.8% | 0.1 | **Production model** |
| Fine-tuned Epoch 6 | 9445 | 86.83% | 0.1 | Overfitting observed |

## Deployment Architecture

### Production Environment
- **Platform**: Red Hat OpenShift AI (RHOAI)
- **Runtime**: vLLM with KServe for high-performance serving
- **Container**: `oci://docker.io/efatnar/modelcar-qwen3-4b-sft:latest`
- **API**: OpenAI-compatible endpoint for seamless integration
- **Monitoring**: Real-time performance metrics and accuracy tracking

### Integration Points
```python
# API Configuration
{
    "endpoint": "https://qwen3-4b-sft-tme-aix.apps.sandbox01.narlabs.io",
    "model": "qwen3-4b-sft",
    "api_version": "v1",
    "authentication": "Bearer token",
    "max_tokens": 50,
    "temperature": 0.1
}
```

## Testing & Quality Assurance

### Test Suite Components
1. **Local Testing** (`test_trained_model.py`)
   - Direct model inference validation
   - Memory usage profiling
   - Response consistency checks

2. **API Testing** (`qa/qa_qwen_intent_classifier.py`)
   - End-to-end integration tests
   - Performance benchmarking
   - Edge case validation
   - Multilingual accuracy assessment

3. **Test Coverage**
   - 70+ intent categories validated
   - 1000+ test cases per evaluation
   - Edge cases: empty queries, special characters, multilingual mixing
   - Performance tests: latency, throughput, concurrent requests

## Lessons Learned & Best Practices

### Success Factors
1. **Systematic Model Evaluation**: Testing 15+ models before selection
2. **Iterative Fine-tuning**: Finding optimal epoch (3) to avoid overfitting
3. **Prompt Engineering**: Structured prompts with clear guidelines
4. **Temperature Optimization**: Low temperature (0.1) for consistency
5. **Infrastructure Choice**: RHOAI providing scalable, enterprise-ready platform

### Challenges Overcome
- **GPU Memory Constraints**: Solved with conservative batch sizes and token limits
- **Multilingual Support**: Achieved 81.44% accuracy for Arabic through data augmentation
- **Storage Requirements**: Managed 100GB+ checkpoint requirements with efficient cleanup
- **Blackwell Compatibility**: Disabled Flash Attention for GPU compatibility

## Future Roadmap

### Short-term Improvements
- [ ] Expand Arabic training data for >90% accuracy
- [ ] Implement active learning for continuous improvement
- [ ] Add real-time feedback loop for model updates
- [ ] Optimize inference for edge deployment

### Long-term Vision
- [ ] Multi-modal support (voice, chat, email)
- [ ] Context-aware conversation flow
- [ ] Personalized intent prediction
- [ ] Integration with knowledge base for automated resolution

## Project Structure

```
intclass/
├── data/
│   └── data.jsonl                # Training dataset (9,445 samples)
├── docker/
│   ├── artifactory.txt           # Container build instructions
│   ├── extract_model.py          # Model extraction utility
│   └── models/                   # Model artifacts directory
├── qa/
│   └── qa_qwen_intent_classifier.py  # Comprehensive QA test suite
├── sft_qwen3-4b-instruct.py     # Fine-tuning training script
├── test_trained_model.py        # Local model testing utility
├── requirements.txt              # Python dependencies
├── CLAUDE.md                     # Development guidelines
└── README.md                     # This document
```

## Quick Start

### Prerequisites
```bash
pip install training_hub requests urllib3
```

### Fine-tuning
```bash
python sft_qwen3-4b-instruct.py \
    --data-path data/data.jsonl \
    --ckpt-output-dir output/ \
    --num-epochs 3
```

### Testing
```bash
# Local testing
python test_trained_model.py

# API testing
python qa/qa_qwen_intent_classifier.py
```

## Contributing

This project is part of the Telco-AIX initiative. For contributions:
1. Review the prompt engineering guidelines
2. Test changes against the full test suite
3. Maintain >90% accuracy threshold
4. Document any infrastructure changes

## Acknowledgments

- **Red Hat OpenShift AI Team**: Infrastructure and platform support
- **Qwen Team**: Base model development
- **Telco Engineering Team**: Domain expertise and validation
- **Open-5G-Hypercore Community**: Collaborative development

## References

- [Prompt Engineering Article](https://medium.com/open-5g-hypercore/episode-xxix-the-prompt-engineering-how-to-make-a-toddler-act-talk-nice-83e9aab2e3b9)
- [Telco-AIX Repository](https://github.com/open-experiments/Telco-AIX)
- [RHOAI Documentation](https://docs.openshift.com/container-platform/)
- [vLLM Performance Guide](https://docs.vllm.ai/)

---

**Project Status**: Production-Ready | **Version**: 1.0 | **Last Updated**: 2025
