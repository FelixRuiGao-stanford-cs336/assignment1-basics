import pickle
from pathlib import Path
from tqdm import tqdm
from .train_bpe import pre_tokenization
import io
import multiprocessing as mp
import os

def pair_position(
    pair: tuple[bytes, bytes],
    pre_token_list: list[bytes]
):
    where = []
    i = 0
    while i < len(pre_token_list)-1:
        if (pre_token_list[i], pre_token_list[i+1]) == pair:
            where.append(i)
            i += 2
        else:
            i += 1

    return where

_global_data = {}

def _init_global_data(vocab_reverse, merges_ranked):
    _global_data["vocab_reverse"] = vocab_reverse
    _global_data["merges_ranked"] = merges_ranked
    
def pre_token_encoder(
    pre_token: bytes,
    vocab_reverse: dict[bytes, int],
    merges_ranked: dict[tuple[bytes, bytes], int]
):
    if pre_token in vocab_reverse:
        return pre_token, [vocab_reverse[pre_token]]

    pre_token_list = [bytes([c]) for c in pre_token]

    while True:
        pair_set = set()
        for i in range(len(pre_token_list)-1):
            pair_set.add((pre_token_list[i], pre_token_list[i+1]))
        new_pre_token_list = []
        best_pair = tuple()
        t = float("inf")
        for pair in pair_set:
            if (rank:=merges_ranked.get(pair, -1)) != -1 and rank<t:
                t = rank
                best_pair = pair
        if not best_pair:
            break
        best_position = set(pair_position(best_pair, pre_token_list))
        i = 0
        while i < len(pre_token_list):
            if i in best_position:
                new_pre_token_list.append(pre_token_list[i]+pre_token_list[i+1])
                i += 2
            else:
                new_pre_token_list.append(pre_token_list[i])
                i += 1
        pre_token_list = new_pre_token_list

    pre_token_id_list = [vocab_reverse[bytes_in_token] for bytes_in_token in pre_token_list]

    return pre_token, pre_token_id_list

def _process_args_encoder(pre_token):
    return pre_token_encoder(pre_token, _global_data["vocab_reverse"], _global_data["merges_ranked"])

def save_pre_tokens_to_disk(
    pre_tokens: list[bytes], 
    temp_dir: str
):
    """
    Save pre-tokens to temporary files on disk.
    Returns the file path of pre-tokens and the set of pre-tokens.
    """

    path_dir = Path(temp_dir)
    path_dir.mkdir(parents=True, exist_ok=True) 
    # Save pre-tokens to temporary file
    file_path = path_dir / "pre-tokens.pkl"

    with open(file_path, 'wb') as f:
        pickle.dump(pre_tokens, f)

    pre_tokens_set = set(pre_tokens)
    return file_path, pre_tokens_set

class Tokenizer:
    def __init__(
        self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] | None = None
):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        if self.special_tokens:
            next_id = len(vocab)
            while next_id in vocab.keys():
                next_id += 1
            for special_token in self.special_tokens:
                b = special_token.encode('utf-8')
                if b in self.vocab.values():
                    continue
                self.vocab[next_id] = b
                next_id += 1
        self.vocab_reverse = {v: k for k,v in self.vocab.items()}
        n = len(merges)
        self.merges_ranked = {}
        for i in range(n):
            self.merges_ranked[self.merges[i]] = i

    @classmethod
    def from_files(
        cls, 
        vocab_filepath: str, 
        merges_filepath: str, 
        special_tokens: list[str] | None = None
):
        with open(vocab_filepath, "rb") as v:
            vocab = pickle.load(v)
        with open(merges_filepath, "rb") as m:
            merges = pickle.load(m)
        return cls(vocab, merges, special_tokens)

    def encode(
        self,
        text: str,
        set_cpus: int = None,
        num_chunks: int = None,
        use_disk_streaming: bool = False
):
        max_cpu = max(1, mp.cpu_count() - 1)
    
        if set_cpus is None:
            num_usecpu = max_cpu
        else:
            num_usecpu = set_cpus if 0 < set_cpus and set_cpus <= max_cpu else max_cpu

        if num_chunks is None:
            actual_num_chunks = num_usecpu  # Default behavior: same as CPU count
        else:
            actual_num_chunks = max(1, num_chunks)

        binary_text = io.BytesIO(text.encode('utf-8'))
        pre_tokens = pre_tokenization(binary_text, self.special_tokens, num_usecpu, actual_num_chunks, use_disk_streaming, False)
        file_path, pre_tokens_set = save_pre_tokens_to_disk(pre_tokens=pre_tokens, temp_dir="pre_tokens")
        del pre_tokens
        
        pre_tokens_vocab = {}
        with mp.Pool(
            processes=num_usecpu,
            initializer=_init_global_data,
            initargs=(self.vocab_reverse, self.merges_ranked)
        ) as pool:
            iterator = pool.imap_unordered(_process_args_encoder, pre_tokens_set, chunksize=128)
            for pre_token, ids in tqdm(iterator, total=len(pre_tokens_set), desc="  Pre-tokenization", unit="chunk"):
                pre_tokens_vocab[pre_token] = ids

        del pre_tokens_set
        with open(file_path, "rb") as f:
            pre_tokens = pickle.load(f)
        pre_tokens_encoded = []
        for pre_token in pre_tokens:
            pre_tokens_encoded.extend(pre_tokens_vocab[pre_token])
        os.remove(file_path)
        return pre_tokens_encoded

