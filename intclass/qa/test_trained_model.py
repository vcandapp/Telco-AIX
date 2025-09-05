#!/usr/bin/env python3
"""
Test the fine-tuned Qwen model for intent classification
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_model():
    # Load the fine-tuned model
    model_path = "output/hf_format/samples_18806/"
    
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"Model loaded successfully!")
    print(f"Model size: {sum(p.numel() for p in model.parameters())/1e9:.1f}B parameters")
    
    # Test cases based on your training data
    test_cases = [
        "follow up on ticket ?",
        "smiles faq please ?", 
        "how do i update to prepaid account ?",
        "What is SWYP and what's the price?",
        "just wanna know when my contract ends ?",
        "would be able to direct me to the service provider stores in my area?",
        "i want to change my sim ?",
        "How do I check my balance?",
        "What are my current charges?"
    ]
    
    print("\n" + "="*60)
    print("Testing Intent Classification")
    print("="*60)
    
    for i, user_query in enumerate(test_cases, 1):
        # Format according to your training template
        prompt = f"""<|im_start|>system
Classify the user question into one of the predefined intents. Respond with only the intent name.<|im_end|>
<|im_start|>user
{user_query}<|im_end|>
<|im_start|>assistant
"""
        
        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                temperature=0.1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the generated part
        generated_text = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        print(f"\nTest {i}:")
        print(f"Query: {user_query}")
        print(f"Intent: {generated_text}")
    
    print("\n" + "="*60)
    print("Testing completed!")
    
    # Memory usage check
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU Memory Used: {memory_used:.2f} GB")

if __name__ == "__main__":
    test_model()