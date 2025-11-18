from cs336_basics.train_bpe import train_bpe
import pickle
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser("BPE training with configurable chunking and disk streaming options.")
    parser.add_argument("--numcpus", "-n", type=int, help="Number of cpu cores to use for parallel processing")
    parser.add_argument("--numchunks", "-c", type=int, help="Number of chunks to divide the file into (default: same as numcpus)")
    parser.add_argument("--disk-streaming", "-d", action="store_true", 
                       help="Use disk-based streaming to reduce memory usage during pre-tokenization")
    args = parser.parse_args()

    input_path = Path("data/TinyStoriesV2-GPT4-train.txt")
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    print(f"Training BPE on {input_path} ...")
    if args.disk_streaming:
        print("Using disk-based streaming for memory-efficient processing")
    
    vocab, merges = train_bpe(
        input_path=str(input_path),
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        cpus=args.numcpus,
        num_chunks=args.numchunks,
        use_disk_streaming=args.disk_streaming
    )

    output_dir = Path("artifacts/tinystories_tokenizer")
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = output_dir / "vocab.pkl"
    merges_path = output_dir / "merges.pkl"

    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)

    print(f"Saved vocab to {vocab_path}")
    print(f"Saved merges to {merges_path}")

if __name__ == "__main__":
    main()
