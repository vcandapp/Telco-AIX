# Telco AI Intent Classification

**Team** : Anil Sonmez, Fatih Nar <br> 
**Credits** : Mustafa Eyceoz, William Caban

## Executive Summary

This project represents a comprehensive transformation of a telco service provider customer support system, replacing fragmented traditional NLP-based intent classifiers with a unified, more-capable Generative AI solution. Through systematic fine-tuning, advanced prompt engineering, and deployment on Red Hat OpenShift AI (RHOAI), we achieved **93.95% accuracy** in intent classification, surpassing the legacy system's performance while providing superior scalability, multilingual capabilities, and conversational context understanding.

## Business Challenge & Strategic Context

### Legacy System Limitations
The customer's existing approach consisted of multiple disconnected components:
- **Fragmented NLP Classifiers**: Individual single-answer intent classifiers lacking conversational context
- **Limited Contextual Understanding**: Traditional NLP struggled with complex customer queries requiring contextual reasoning
- **Dual-System Complexity**: Mix of proprietary NLP systems and OpenAI ChatGPT API consumption
- **Compliance & Cost Issues**: External AI services posed data sovereignty concerns and escalating operational costs
- **Language Mismatch**: English-centric models underperformed in Arabic-primary regions

### Strategic Business Drivers
The customer sought to migrate from isolated NLP components to **integrated solutions built around large language models** for several critical reasons:

1. **AI Sovereignty & Compliance**: On-premise deployment ensuring local data regulation compliance
2. **Cost Optimization**: Superior price/performance through benchmarked fine-tuned models vs. AIaaS consumption
3. **Unified Conversational Context**: Single LLM-based architecture enabling contextual understanding across customer interactions
4. **Regional Language Priority**: Arabic-first approach with English as supporting capability
5. **Operational Security**: Distributed small infrastructure deployment maintaining data control

## Solution Architecture

### Integrated LLM Architecture
- **Base Model**: Qwen3-4B-Instruct (fine-tuned for telco domain)
- **Infrastructure**: Red Hat OpenShift AI (RHOAI) on-premise
- **Serving Runtime**: vLLM with KServe for high-performance inference
- **Training Framework**: Supervised Fine-Tuning (SFT) with training_hub
- **Deployment**: Distributed small infrastructure for optimal price/performance

### Key Architectural Advantages
- **Unified Context Processing**: Single model handles conversational flow vs. fragmented classifiers
- **Multi-Modal Capability**: Beyond classification to general question answering
- **Cost-Effective Scaling**: On-premise vLLM+KServe runtime eliminates per-token API costs
- **Data Sovereignty**: Complete control over sensitive customer communication data
- **Language Flexibility**: Arabic-English translation pipeline for optimal accuracy

## Technical Implementation

### 1. Model Selection & Evaluation Journey

Our systematic evaluation process tested multiple models to identify the optimal solution:

| Model | Parameters | Accuracy | Platform | Notes |
|-------|------------|----------|----------------|-------|
| Granite-7B-Lab | 7B | 19.96% | RHOAI | Baseline test |
| Granite-8B-Lab-v1 | 8B | 41.46% | RHOAI | With context definitions |
| Llama-3.2-3B | 3B | 49.05% | RHOAI | Lightweight option |
| Phi-4 | - | 78.62% | RHOAI | Microsoft model |
| Llama-4-Scout-17B | 17B | 83.32% | RHOAI | Best pre-trained performance |
| Qwen3-32B (Full) | 32B | 87.10% | RHOAI | Large model baseline |
| **Qwen3-4B (Fine-tuned)** | **4B** | **93.22%** | **RHOAI** | **v1 on Faster Learning Rate + 3 Epochs** |
| **Qwen3-4B (Fine-tuned)** | **4B** | **93.95%** | **RHOAI** | **v2 on Slower Learning Rate + 10 Epochs** |

### 2. Fine-Tuning Process

#### Dataset Preparation
- **Training Samples**: 9,445 conversation examples
- **Format**: JSONL with structured conversation format
- **Languages**: Arabic (primary) and English (supporting)
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
- **Training Time**: ~15 minutes per epoch

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

## Beyond Classification: General Question Answering

### Extended Capabilities
While the primary focus is intent classification, the Qwen3-4B-Instruct model provides comprehensive conversational AI capabilities:

**Natural Language Understanding**
- Complex query interpretation with contextual awareness
- Multi-turn conversation handling with memory retention
- Nuanced understanding of customer emotions and urgency levels

**General Knowledge & Domain Expertise**
- Telco industry knowledge for technical explanations
- Service plan details and feature comparisons
- Troubleshooting guidance beyond simple classification
- Regulatory and policy information delivery

**Conversational Flow Management**
- Dynamic response generation based on customer context
- Escalation path recommendations
- Follow-up question suggestions for complete issue resolution
- Seamless handover to human agents with complete context preservation

This unified approach eliminates the need for separate systems handling classification vs. conversation, providing a cohesive customer experience through a single, context-aware AI agent.

## Results & Performance

### Accuracy Achievements vs. Expectations
```
Customer Expectations:        90%+ English, 80%+ Arabic
Current Results:
  English:                    93.95% ✓ (Exceeds target)
  Arabic:                     81.44% ✓ (Meets target)
Legacy NLP System:            ~75% accuracy
Initial LLM Tests:            38-83% (various models)
```

### Performance Optimization Strategy
**Ongoing Accuracy Improvements**: We are actively working on enhanced training datasets featuring:
- **Higher Volume**: Expanding from 9,445 to 25,000+ conversation examples
- **Greater Variety**: Incorporating regional dialects, edge cases, and complex scenarios
- **Advanced Fine-tuning**: Iterative model refinement with domain-specific examples
- **Enhanced Prompt Engineering**: Continuous optimization of classification guidelines

### Key Performance Metrics
- **Response Time**: <500ms average inference
- **Throughput**: 1000+ requests/minute
- **Model Size**: 4B parameters (optimized for edge deployment)
- **Memory Usage**: ~8-16GB VRAM
- **Multilingual Support**: Arabic (primary), English (supporting)

### Language Strategy: Arabic-First Approach
**Translation Pipeline Approach**: For optimal accuracy in Arabic-primary regions:
1. **Direct Arabic Processing**: Native Arabic intent classification for straightforward queries
2. **Translation Enhancement**: Arabic ↔ English translation pipeline for complex technical terms
3. **Hybrid Processing**: Leveraging English model strength while maintaining Arabic primacy
4. **Cultural Context**: Regional dialect recognition and culturally appropriate responses

## Deployment Architecture

### Production Environment
- **Platform**: Red Hat OpenShift AI (RHOAI) on-premise
- **Runtime**: vLLM with KServe for high-performance serving
- **Container**: `oci://docker.io/efatnar/modelcar-qwen3-4b-sft:latest`
- **API**: OpenAI-compatible endpoint for seamless integration
- **Monitoring**: Real-time performance metrics and accuracy tracking
- **Infrastructure**: Distributed small-scale deployment for optimal price/performance

### Deployment Configurations
```
# vLLM Configuration
qwen3-4b-sft-v1
oci://docker.io/efatnar/modelcar-qwen3-4b-sft:v1
--max-model-len=32768
--gpu-memory-utilization=0.95
---
qwen3-4b-sft-v2
oci://docker.io/efatnar/modelcar-qwen3-4b-sft:v2
--max-model-len=32768
--gpu-memory-utilization=0.95
```

### Cost & Security Benefits
- **Price/Performance**: On-premise deployment eliminates per-token API costs
- **Data Sovereignty**: Complete control over customer data and AI processing
- **Regulatory Compliance**: Meets local data protection and AI governance requirements
- **Scalable Infrastructure**: Distributed deployment scales with business growth
- **Operational Control**: Full visibility and control over model behavior and updates

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
3. **Prompt Engineering**: Structured prompts with clear guidelines. [See Examples Here](https://github.com/open-experiments/Telco-AIX/blob/main/telco-sme/system_prompts.json)
4. **Temperature Optimization**: Low temperature (0.1) for consistency
5. **Infrastructure Choice**: RHOAI providing scalable, enterprise-ready platform
6. **Business Alignment**: Addressing AI sovereignty and cost concerns upfront

### Challenges Overcome
- **GPU Memory Constraints**: Solved with conservative batch sizes and token limits
- **Multilingual Support**: Achieved 81.44% accuracy for Arabic through data augmentation
- **Storage Requirements**: Managed 100GB+ checkpoint requirements with efficient cleanup
- **Blackwell Compatibility**: Disabled Flash Attention for GPU compatibility
- **Migration Complexity**: Seamless transition from fragmented NLP to unified LLM architecture

## Future Roadmap

### Short-term Improvements (Q1-Q2)
- [ ] Expand Arabic training data for >85% accuracy
- [ ] Implement active learning for continuous improvement
- [ ] Add real-time feedback loop for model updates
- [ ] Optimize inference for edge deployment
- [ ] Enhanced translation pipeline for technical terminology

### Long-term Vision (Q3-Q4)
- [ ] Multi-modal support (voice, chat, email)
- [ ] Context-aware conversation flow with memory
- [ ] Personalized intent prediction based on customer history
- [ ] Integration with knowledge base for automated resolution
- [ ] Advanced Arabic dialect recognition and processing

## Project Structure

```
intclass/
├── data/
│   └── data.jsonl                # Training dataset (put your data here)
├── docker/
│   ├── artifactory.txt           # Container build instructions
│   ├── extract_model.py          # Model extraction utility
│   └── models/                   # Model artifacts directory
├── qa/
│   └── qa_qwen_intent_classifier.py  # Comprehensive QA test suite
│   └── test_trained_model.py     # Local model testing utility
├── sft_qwen3-4b-instruct.py     # Fine-tuning training script
├── requirements.txt              # Python dependencies
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
python qa/test_trained_model.py

# API testing
python qa/qa_qwen_intent_classifier.py
```

## References

- [Prompt Engineering Article](https://medium.com/open-5g-hypercore/episode-xxix-the-prompt-engineering-how-to-make-a-toddler-act-talk-nice-83e9aab2e3b9)
- [Training Hub](https://github.com/Red-Hat-AI-Innovation-Team/training_hub)
- [Telco-SME Prompt UI & AI Observability](https://github.com/open-experiments/Telco-AIX/tree/main/telco-sme)
- [Qwen3-4B-Instruct Model Details](https://huggingface.co/Qwen/Qwen3-4B-Instruct)
