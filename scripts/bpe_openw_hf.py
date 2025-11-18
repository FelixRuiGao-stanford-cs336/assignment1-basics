"""
HuggingFace Tokenizers version for BPE training speed comparison.
Uses Rust-based implementation for significantly faster training.
"""

from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from pathlib import Path
import argparse
import time


def main():
    parser = argparse.ArgumentParser("BPE training using HuggingFace Tokenizers (Rust-based)")
    parser.add_argument("--show-progress", "-p", action="store_true", 
                       help="Show training progress bar")
    args = parser.parse_args()

    input_path = Path("data/owt_train.txt")
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]

    print("="*60)
    print("🚀 Training BPE with HuggingFace Tokenizers (Rust)")
    print(f"Input file: {input_path}")
    print(f"Vocab size: {vocab_size}")
    print(f"Special tokens: {special_tokens}")
    print("="*60)
    
    # Initialize BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token=None))
    
    # Use GPT-2 style byte-level pre-tokenization
    # Note: This is slightly different from the original regex pattern, but:
    # 1. It's the standard approach used by GPT-2/GPT-3
    # 2. Works at the byte level, handling all Unicode properly
    # 3. Much faster and more robust
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # Configure trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=args.show_progress,
        min_frequency=0  # No frequency threshold
    )
    
    # Start timing
    start_time = time.time()
    
    # Train on the file
    print("\n[Training Started...]")
    tokenizer.train([str(input_path)], trainer)
    
    # End timing
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n✅ Training completed!")
    print(f"⏱️  Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    # Save the tokenizer
    output_dir = Path("artifacts/openwebtext_tokenizer_hf")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    
    print(f"\n💾 Saved tokenizer to {tokenizer_path}")
    
    # Display vocab info
    vocab = tokenizer.get_vocab()
    print(f"📊 Final vocab size: {len(vocab)}")
    
    # Test tokenization
    test_text = "Hello, world! This is a test.<|endoftext|>"
    encoded = tokenizer.encode(test_text)
    print(f"\n🧪 Test encoding: '{test_text}'")
    print(f"   Token IDs: {encoded.ids}")
    print(f"   Tokens: {encoded.tokens}")
    
    print("\n" + "="*60)
    print("Done! 🎉")
    print("="*60)


if __name__ == "__main__":
    main()