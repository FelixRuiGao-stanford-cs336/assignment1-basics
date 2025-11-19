import os
from collections import Counter, defaultdict
from typing import Any, BinaryIO
import multiprocessing as mp
import regex as re
import time
import logging
from datetime import datetime
import tempfile
import shutil
import pickle
from tqdm import tqdm

def _process_chunks_star(args):
    """
    Small wrapper to unpack (chunk, special_tokens) for imap_unordered.
    Keeping process_chunks() signature unchanged.
    """
    chunk, special_tokens = args
    return process_chunks(chunk, special_tokens)

def _process_chunk_file_and_cleanup(args):
    """
    Wrapper for disk-streaming: process a chunk file and remove it immediately.
    This enables streaming updates + early cleanup while keeping the original
    process_chunk_from_file() unchanged.
    """
    chunk_file_path, special_tokens = args
    try:
        return process_chunk_from_file(chunk_file_path, special_tokens)
    finally:
        # Best-effort cleanup; ignore failures (e.g., already removed)
        try:
            os.remove(chunk_file_path)
        except OSError:
            pass
# Pre-Tokenization:

def divide_chunks(
    file: BinaryIO,
    num_chunks: int,
    special_token: bytes):

    file.seek(0, os.SEEK_END)
    size = file.tell()

    non_zero_boundaries = [size*(i+1)//num_chunks for i in range(num_chunks)]
    non_zero_boundaries[-1] = size

    final_boundaries = {0, size}
    for boundary in non_zero_boundaries:
        if(boundary == size):
            break

        # Read the -4k and 4k range to find special tokens
        start_position = max(boundary-4096, 0)
        end_position = min(boundary+4096, size)
        file.seek(start_position)
        mini_chunk = file.read(end_position-start_position)

        # Search to the right first to avoid an uneven division when the file size is small
        if not mini_chunk: # Anyway, it should not be None
            continue
        rel_position = mini_chunk.find(special_token, boundary-start_position)
        if rel_position == -1:
            rel_position = mini_chunk.find(special_token, 0, boundary-start_position)
            if rel_position == -1:
                continue

        final_boundaries.add(start_position+rel_position)

    return sorted(list(final_boundaries))
    
    
def process_chunks(
    chunk: bytes,
    special_tokens: list[str]
):
    chunk_string = chunk.decode('utf-8', errors="ignore")
    pat_special_token = re.compile("|".join(re.escape(special_token) for special_token in special_tokens))
    text_split = [split_chunk for split_chunk in re.split(pat_special_token, chunk_string) if split_chunk]
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    pre_tokens_chunk = Counter()
    for minichunk in text_split:
        matches = re.finditer(PAT, minichunk)
        for match in matches:
            pre_token_int = tuple(match.group().encode('utf-8'))
            pre_tokens_chunk[pre_token_int] += 1
    return pre_tokens_chunk


def process_chunk_from_file(
    chunk_file_path: str,
    special_tokens: list[str]
):
    """
    Process a chunk that was saved to disk.
    This is a wrapper function for disk-based streaming processing.
    """
    with open(chunk_file_path, 'rb') as f:
        chunk = f.read()
    return process_chunks(chunk, special_tokens)


def save_chunks_to_disk(
    file: BinaryIO,
    chunk_boundaries: list[int],
    temp_dir: str
) -> list[str]:
    """
    Save chunks to temporary files on disk.
    Returns a list of file paths for the saved chunks.
    """
    chunk_files = []
    for i in range(len(chunk_boundaries)-1):
        file.seek(chunk_boundaries[i])
        chunk_data = file.read(chunk_boundaries[i+1]-chunk_boundaries[i])
        
        # Save chunk to temporary file
        chunk_file_path = os.path.join(temp_dir, f"chunk_{i}.bin")
        with open(chunk_file_path, 'wb') as chunk_file:
            chunk_file.write(chunk_data)
        chunk_files.append(chunk_file_path)
        
        logging.debug(f"  Saved chunk {i+1}/{len(chunk_boundaries)-1} to disk ({len(chunk_data)} bytes)")
    
    return chunk_files


def pre_tokenization_disk_streaming(
    file: BinaryIO,
    special_tokens: list[str],
    num_chunks: int,
    num_cpus: int
):
    """
    Disk-based streaming version of pre-tokenization.
    Saves chunks to disk first, then processes them with streaming updates.
    This reduces memory usage by not keeping all chunks in memory simultaneously.
    """
    logging.info("  Using disk-based streaming for pre-tokenization...")
    
    # Create temporary directory for chunks
    temp_dir = tempfile.mkdtemp(prefix="bpe_chunks_")
    logging.info(f"  Created temporary directory: {temp_dir}")
    
    try:
        # Divide file into chunks
        chunk_boundaries = divide_chunks(file, num_chunks, b'<|endoftext|>')
        logging.info(f"  Divided file into {len(chunk_boundaries)-1} chunks")
        
        # Save chunks to disk
        chunk_files = save_chunks_to_disk(file, chunk_boundaries, temp_dir)
        logging.info(f"  Saved all chunks to disk")
        
        pre_tokens = Counter()
        
        if num_cpus == 1:
            # Process chunks sequentially with streaming updates
            logging.info("  Processing chunks sequentially (1 CPU)...")
            for chunk_file in tqdm(chunk_files, desc="  Pre-tokenization", unit="chunk"):
                chunk_pre_tokens = process_chunk_from_file(chunk_file, special_tokens)
                pre_tokens.update(chunk_pre_tokens)
                # Delete the chunk file immediately after processing to save space
                try:
                    os.remove(chunk_file)
                except OSError:
                    pass
        else:
            # Process chunks in parallel with streaming updates
            logging.info(f"  Processing chunks in parallel ({num_cpus} CPUs) with streaming updates...")
            iteration_jobs = [(chunk_file, special_tokens) for chunk_file in chunk_files]
    
            # Stream results as they complete and update pre_tokens immediately
            with mp.Pool(processes=num_cpus) as pool:
                for chunk_result in tqdm(
                    pool.imap_unordered(_process_chunk_file_and_cleanup, iteration_jobs),
                    total=len(iteration_jobs),
                    desc="  Pre-tokenization",
                    unit="chunk"
                ):
                    pre_tokens.update(chunk_result)
        
        logging.info(f"  Disk-based streaming completed successfully")
        
    finally:
        # Clean up temporary directory (best-effort; files should already be removed)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logging.info(f"  Cleaned up temporary directory: {temp_dir}")
    
    return pre_tokens


def pre_tokenization(
    file: BinaryIO,
    special_tokens: list[str],
    set_cpus: int = None,
    num_chunks: int = None,
    use_disk_streaming: bool = False
):
    """
    Pre-tokenization with support for both in-memory and disk-based streaming processing.

    Changes:
      - In the multi-process path, switch to imap_unordered() so each chunk's result
        updates `pre_tokens` immediately (streaming merge), instead of collecting
        all per-chunk Counters and merging afterwards.
    
    Args:
        file: Input file to tokenize
        special_tokens: List of special tokens
        set_cpus: Number of CPUs to use for parallel processing (default: mp.cpu_count()-1)
        num_chunks: Number of chunks to divide the file into (default: same as set_cpus)
        use_disk_streaming: If True, use disk-based streaming to reduce memory usage
    """
    max_cpu = max(1, mp.cpu_count() - 1)
    
    # Determine number of CPUs to use
    if set_cpus is None:
        num_usecpu = max_cpu
    else:
        num_usecpu = set_cpus if set_cpus <= max_cpu else max_cpu
    
    # Determine number of chunks
    if num_chunks is None:
        actual_num_chunks = num_usecpu  # Default behavior: same as CPU count
    else:
        actual_num_chunks = max(1, num_chunks)
    
    logging.info(f"  Number of chunks: {actual_num_chunks}")
    logging.info(f"  Number of CPUs: {num_usecpu}")
    logging.info(f"  Disk streaming: {'Enabled' if use_disk_streaming else 'Disabled'}")
    
    # If disk streaming is enabled, use the disk-based version
    if use_disk_streaming:
        return pre_tokenization_disk_streaming(file, special_tokens, actual_num_chunks, num_usecpu)
    
    # In-memory processing logic
    chunk_boundaries = divide_chunks(file, actual_num_chunks, b'<|endoftext|>')
    chunks = []
    pre_tokens = Counter()
    
    for i in range(len(chunk_boundaries)-1):
        file.seek(chunk_boundaries[i])
        chunks.append(file.read(chunk_boundaries[i+1]-chunk_boundaries[i]))

    if num_usecpu == 1:
        logging.info("  Processing chunk with progress bar...")
        for chunk in tqdm(chunks, desc="  Pre-tokenization", unit="chunk"):
            chunk_result = process_chunks(chunk=chunk, special_tokens=special_tokens)
            pre_tokens.update(chunk_result)
        return pre_tokens

    # Multi-process: stream results and update pre_tokens immediately
    iteration_jobs = [(chunk, special_tokens) for chunk in chunks]
    logging.info(f"  Processing chunks in parallel ({num_usecpu} CPUs) with streaming updates...")
    with mp.Pool(processes=num_usecpu) as pool:
        for pre_tokens_by_chunk in tqdm(
            pool.imap_unordered(_process_chunks_star, iteration_jobs),
            total=len(iteration_jobs),
            desc="  Pre-tokenization",
            unit="chunk"
        ):
            pre_tokens.update(pre_tokens_by_chunk)
        
    return pre_tokens
    

# BPE:

def find_pair(
    pair: tuple[int],
    token: tuple[int]
):
    where = []
    if not pair or not token or len(token) <= 1:
        return where
    i = 0
    while i <= len(token)-2:
        if pair == token[i:(i+2)]:
            where.append(i)
            i+=2
        else:
            i+=1
    return where


def reconstruct_token(
    where: list[int],
    token: tuple[int],
    new_id
):
    new_token = ()
    i = 0
    while i <= len(token)-1:
        if i in where:
            new_token = new_token + (new_id,)
            i += 2
        else:
            new_token = new_token + (token[i],)
            i += 1
    
    return new_token
    

def get_best_pair(
    pair_counts: dict[tuple[int, int], int], 
    vocab: dict[int, tuple[int]]
):
    if not pair_counts:
        return None
    return max(pair_counts, key = lambda pair: (pair_counts[pair], vocab[pair[0]], vocab[pair[1]]))


def setup_logging(log_file=None):
    if log_file is None:
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        # Generate log filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'logs/bpe_training_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    cpus: int = None,
    num_chunks: int = None,
    use_disk_streaming: bool = False
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train BPE tokenizer with support for both in-memory and disk-based streaming processing.
    
    Args:
        input_path: Path to the input text file
        vocab_size: Target vocabulary size
        special_tokens: List of special tokens to include
        cpus: Number of CPUs to use for parallel processing (default: mp.cpu_count()-1)
        num_chunks: Number of chunks to divide the file into (default: same as cpus)
        use_disk_streaming: If True, use disk-based streaming to reduce memory usage
    
    Returns:
        Tuple of (vocab dict, merges list)
    """

    if not logging.getLogger().handlers:
        setup_logging()
    
    logging.info("="*60)
    logging.info(f"Starting BPE tokenizer training")
    logging.info(f"Input file: {input_path}")
    logging.info(f"Target vocab size: {vocab_size}")
    logging.info(f"Special tokens: {special_tokens}")
    logging.info(f"Disk streaming: {'Enabled' if use_disk_streaming else 'Disabled'}")
    logging.info("="*60)
    
    logging.info("\n[1/2] Starting Pre-tokenization phase...")
    pre_tokenization_start = time.time()


    with open(input_path, "rb") as f:
        pre_tokens = pre_tokenization(
            f, 
            special_tokens, 
            set_cpus=cpus,
            num_chunks=num_chunks,
            use_disk_streaming=use_disk_streaming
        )


    pre_tokenization_end = time.time()
    pre_tokenization_time = pre_tokenization_end - pre_tokenization_start
    
    logging.info(f" Pre-tokenization completed!")
    logging.info(f"  Time elapsed: {pre_tokenization_time:.2f} seconds ({pre_tokenization_time/60:.2f} minutes)")
    logging.info(f"  Number of pre-tokens: {len(pre_tokens)}")
    logging.info(f"  Total token count: {sum(pre_tokens.values())}")
    
    logging.info("\n[2/2] Starting BPE Training phase...")
    bpe_start = time.time()


    vocab = defaultdict(bytes)
    merges = []
    for i in range(256):
        vocab[i] = bytes([i])

    next_id = 256
    for special_token in special_tokens:
        vocab[next_id] = special_token.encode('utf-8')
        next_id += 1
    if next_id >= vocab_size:


        bpe_end = time.time()
        bpe_time = bpe_end - bpe_start
        logging.info(f" BPE Training completed! (Early termination: special tokens reached vocab size)")
        logging.info(f"  Time elapsed: {bpe_time:.2f} seconds")
        
        total_time = pre_tokenization_time + bpe_time
        logging.info("\n" + "="*60)
        logging.info("Training completed! Summary:")
        logging.info(f"  Pre-tokenization: {pre_tokenization_time:.2f} seconds ({pre_tokenization_time/total_time*100:.1f}%)")
        logging.info(f"  BPE Training:     {bpe_time:.2f} seconds ({bpe_time/total_time*100:.1f}%)")
        logging.info(f"  Total time:       {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        logging.info("="*60)

        
        return vocab, merges

    pair_counts = Counter()
    pair_positions = defaultdict(set)

    for pre_token_k, pre_token_v in pre_tokens.items():
        for i in range(len(pre_token_k)-1):
            pair = (pre_token_k[i], pre_token_k[i+1])
            pair_counts[pair] += pre_token_v
            pair_positions[pair].add(pre_token_k)

    merge_count = 0
    total_merges = vocab_size - next_id
    last_log_time = time.time()

    
    while next_id < vocab_size:
        best_pair = get_best_pair(pair_counts=pair_counts, vocab=vocab)
        vocab[next_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        
        merge_count += 1
        
        current_time = time.time()
        if current_time - last_log_time >= 10 or merge_count % 1000 == 0:
            elapsed = current_time - bpe_start
            progress = merge_count / total_merges * 100
            logging.info(f"  Progress: {merge_count}/{total_merges} ({progress:.1f}%) - "
                        f"Elapsed time: {elapsed:.1f}s")
            last_log_time = current_time
        
        if pair_counts[best_pair] <= 0:
            bpe_end = time.time()
            bpe_time = bpe_end - bpe_start
            logging.info(f"\n BPE Training completed! (Early termination: pair count <= 0)")
            logging.info(f"  Time elapsed: {bpe_time:.2f} seconds ({bpe_time/60:.2f} minutes)")
            logging.info(f"  Number of merges: {merge_count}")
            
            total_time = pre_tokenization_time + bpe_time
            logging.info("\n" + "="*60)
            logging.info("Training completed! Summary:")
            logging.info(f"  Pre-tokenization: {pre_tokenization_time:.2f} seconds ({pre_tokenization_time/total_time*100:.1f}%)")
            logging.info(f"  BPE Training:     {bpe_time:.2f} seconds ({bpe_time/total_time*100:.1f}%)")
            logging.info(f"  Total time:       {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            logging.info("="*60)
            
            return vocab, merges
            
        positions = list(pair_positions[best_pair])
        for pos in positions:
            where_token = find_pair(best_pair, pos)
            if not where_token:
                continue

            # Delete the footprint of this pre-token
            for i in range(len(pos)-1):
                pair = (pos[i], pos[i+1])
                pair_counts[pair] -= pre_tokens[pos]
                if pair_counts[pair] < 0:
                    raise ValueError("Pair counts should not be negative!")
                if pos in pair_positions[pair]:
                    pair_positions[pair].remove(pos)
            
            # Add the pre-token with the merged pair
            pre_token_new = reconstruct_token(where_token, pos, next_id)
            
            for i in range(len(pre_token_new)-1):
                pair = (pre_token_new[i], pre_token_new[i+1])
                pair_counts[pair] += pre_tokens[pos]
                pair_positions[pair].add(pre_token_new)
            if pre_token_new in pre_tokens:
                raise ValueError("There should not be new pre-token before we merge the new pair!")
            pre_tokens[pre_token_new] = pre_tokens[pos]

            del pre_tokens[pos]
        
        del pair_counts[best_pair], pair_positions[best_pair]
        next_id += 1
    
    bpe_end = time.time()
    bpe_time = bpe_end - bpe_start
    
    logging.info(f"\n BPE Training completed!")
    logging.info(f"  Time elapsed: {bpe_time:.2f} seconds ({bpe_time/60:.2f} minutes)")
    logging.info(f"  Number of merges: {merge_count}")
    
    total_time = pre_tokenization_time + bpe_time
    logging.info("\n" + "="*60)
    logging.info("Training completed! Summary:")
    logging.info(f"  Pre-tokenization: {pre_tokenization_time:.2f} seconds ({pre_tokenization_time/total_time*100:.1f}%)")
    logging.info(f"  BPE Training:     {bpe_time:.2f} seconds ({bpe_time/total_time*100:.1f}%)")
    logging.info(f"  Total time:       {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    logging.info("="*60 + "\n")
    
    return vocab, merges

if __name__ == "__main__":
    setup_logging()
