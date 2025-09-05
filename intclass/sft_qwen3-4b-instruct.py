#!/usr/bin/env python3
"""
SFT Training Example: Qwen/Qwen3-4B-Instruct-2507 (Single GPU)
This script demonstrates SFT training with Qwen/Qwen3-4B-Instruct-2507 model
using a single GPU setup with training_hub, optimized for RTX PRO 6000 Blackwell.

Prereq: 
	pip install training_hub

Example usage:
    python sft_qwen3-4b-instruct.py --data-path data/cic.jsonl --ckpt-output-dir output/
"""
import os
import sys
import time
from datetime import datetime
import argparse
from training_hub import sft


def main():
    parser = argparse.ArgumentParser(description='SFT Training Example: Qwen3-4B-Instruct-2507 (Single GPU)')
    
    # Required parameters
    parser.add_argument('--data-path', required=True,
                       help='Path to training data (JSONL format)')
    parser.add_argument('--ckpt-output-dir', required=True,
                       help='Directory to save checkpoints')
    
    # Optional overrides
    parser.add_argument('--model-path', default='Qwen/Qwen3-4B-Instruct-2507',
                       help='Model path or HuggingFace name (default: Qwen/Qwen3-4B-Instruct-2507)')
    parser.add_argument('--num-epochs', type=int, default=10,
                       help='Number of epochs (default: 10)')
    parser.add_argument('--max-tokens-per-gpu', type=int, default=8192,
                       help='Max tokens per GPU (default: 8192, reduced for 4B model)')
    parser.add_argument('--nproc-per-node', type=int, default=1,
                       help='Number of GPUs (default: 1 for single GPU)')
    parser.add_argument('--effective-batch-size', type=int, default=16,
                       help='Effective batch size (default: 16 for single GPU)')
    parser.add_argument('--learning-rate', type=float, default=2e-6,
                       help='Learning rate (default: 2e-6, lower for 4B model)')
    parser.add_argument('--max-seq-len', type=int, default=8192,
                       help='Max sequence length (default: 8192, reduced for memory)')
    
    args = parser.parse_args()
    
    # Validate parameter compatibility
    if args.max_tokens_per_gpu < args.max_seq_len:
        print(f"ERROR: max_tokens_per_gpu ({args.max_tokens_per_gpu}) must be >= max_seq_len ({args.max_seq_len})")
        print("Auto-adjusting max_tokens_per_gpu to match max_seq_len...")
        args.max_tokens_per_gpu = args.max_seq_len + 1000  # Add buffer
        print(f"Updated max_tokens_per_gpu to: {args.max_tokens_per_gpu}")
        print()
    
    # Validate single GPU setup
    if args.nproc_per_node > 1:
        print("WARNING: Multiple GPUs specified but this script is optimized for single GPU.")
        print("Setting nproc_per_node=1 for single GPU training.")
        args.nproc_per_node = 1
    
    # Configuration summary
    print("🚀 SFT Training: Qwen3-4b-instruct (Single GPU)")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.ckpt_output_dir}")
    print(f"GPUs: {args.nproc_per_node}")
    print(f"Max tokens per GPU: {args.max_tokens_per_gpu:,}")
    print(f"Effective batch size: {args.effective_batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max sequence length: {args.max_seq_len}")
    print(f"Epochs: {args.num_epochs}")
    print()
    print("💡 Optimizations for RTX PRO 6000 Blackwell:")
    print("   ✓ Flash Attention disabled (Blackwell compatibility)")
    print("   ✓ Reduced batch size for 4B model single GPU training")
    print("   ✓ Conservative memory settings to avoid OOM")
    print("   ✓ Lower learning rate for large model stability")
    print()
    
    # Training configuration optimized for single RTX PRO 6000 + 30B model
    start_time = time.time()
    
    try:
        result = sft(
            # Model and data
            model_path=args.model_path,
            data_path=args.data_path,
            ckpt_output_dir=args.ckpt_output_dir,
            
            # CRITICAL: Disable Flash Attention for Blackwell compatibility
            disable_flash_attn=True,
            
            # Training parameters optimized for single GPU + 30B model
            num_epochs=args.num_epochs,
            effective_batch_size=args.effective_batch_size,  # Reduced from 128 for single GPU
            learning_rate=args.learning_rate,               # Lower LR for stability with 30B model
            max_seq_len=args.max_seq_len,                   # Reduced from 16384 to save memory
            max_tokens_per_gpu=args.max_tokens_per_gpu,     # Conservative for 30B model
            
            # Memory optimization
            data_output_dir="/dev/shm",                     # Use RAM disk for speed
            warmup_steps=50,                                # Reduced warmup steps
            save_samples=0,                                 # Disable sample-based checkpointing
            
            # Checkpointing strategy
            checkpoint_at_epoch=True,                       # Save at each epoch
            accelerate_full_state_at_epoch=True,            # Full state for resumption
            
            # Single GPU setup (CRITICAL CHANGE)
            nproc_per_node=1,                              # Single GPU
            nnodes=1,                                      # Single node
            node_rank=0,                                   # Primary node
            rdzv_id=100,                                   # Rendezvous ID
            rdzv_endpoint="127.0.0.1:29500",              # Local endpoint
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("=" * 60)
        print("✅ Training completed successfully!")
        print(f"⏱️  Duration: {duration/3600:.2f} hours")
        print(f"📁 Checkpoints: {args.ckpt_output_dir}/hf_format/")
        print()
        print("🎯 Next Steps:")
        print("   • Check checkpoint quality")
        print("   • Run inference tests")
        print("   • Consider further fine-tuning if needed")
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print("=" * 60)
        print(f"❌ Training failed after {duration/60:.1f} minutes")
        print(f"Error: {e}")
        print()
        print("🔧 Troubleshooting suggestions:")
        if "out of memory" in str(e).lower() or "oom" in str(e).lower():
            print("   💾 Memory issue detected:")
            print(f"      • Try --max-tokens-per-gpu {args.max_tokens_per_gpu//2}")
            print(f"      • Try --effective-batch-size {max(1, args.effective_batch_size//2)}")
            print(f"      • Try --max-seq-len {args.max_seq_len//2}")
            print("      • Consider using a smaller model (1B or 2B)")
        elif "invalid device" in str(e).lower():
            print("   🖥️  GPU configuration issue:")
            print("      • Verify GPU is visible with: nvidia-smi")
            print("      • Check CUDA installation")
            print("      • Ensure PyTorch detects GPU")
        else:
            print("   🔍 General troubleshooting:")
            print("      • Check data format and paths")
            print("      • Verify model name/path is correct")
            print("      • Review training logs above")
        
        sys.exit(1)


if __name__ == "__main__":
    main()