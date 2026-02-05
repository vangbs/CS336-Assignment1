import os
import pickle
from typing import Iterable, Iterator
import regex as re

class BPE_tokenizer:
    def __init__(
        self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.merge_id = {merge: i for i, merge in enumerate(self.merges)}
    
    @classmethod
    def from_files(
        cls, 
        vocab_merges_filepath: str | os.PathLike, 
        special_tokens: list[str]
    ):
        with open(vocab_merges_filepath, "rb") as f:
            data = pickle.load(f)
        return cls(data["vocab"], data["merges"], special_tokens)
    
    def encode(self, text: str) -> list[int]:
        return list(self.encode_iterable([text]))
    
    def encode_chunk(self, text: str) -> Iterator[int]:
        pretoken_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for a in re.finditer(pretoken_pattern, text):
            pretoken = a.group().encode("utf-8")
            byte_list: list[bytes] = [pretoken[i:i+1] for i in range(len(pretoken))]
            while True:
                bp_list: list[tuple[int, tuple[bytes, bytes]]] = []
                for i in range(len(byte_list) - 1):
                    bp = (byte_list[i], byte_list[i+1])
                    if bp in self.merge_id:
                        bp_list.append((self.merge_id[bp], bp))
                if not bp_list:
                    break
                min_bp = min(bp_list)[1]
                curr_list: list[bytes] = []
                index = 0
                while index < len(byte_list):
                    if index + 1 < len(byte_list) and (byte_list[index], byte_list[index + 1]) == min_bp:
                        curr_list.append(min_bp[0] + min_bp[1])
                        index += 2
                    else:
                        curr_list.append(byte_list[index])
                        index += 1
                byte_list = curr_list
            for b in byte_list:
                yield self.reverse_vocab[b]
        
            
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        if self.special_tokens:
            # Prevent that some special_tokens are prefix of others.
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = "|".join(re.escape(t) for t in sorted_tokens)
            # Do not need to consider the case that pretokens crossing the border of s.
            for s in iterable:
                curr_start: int = 0
                # Do not need to consider the case that special tokens crossing the border of s, because they don't contain '\n'.
                for a in re.finditer(special_pattern, s):
                    yield from self.encode_chunk(s[curr_start:a.start()])
                    yield self.reverse_vocab[a.group().encode("utf-8")]
                    curr_start = a.end()
                yield from self.encode_chunk(s[curr_start:])
        else:
            # Do not need to consider the case that pretokens crossing the border of s.
            for s in iterable:
                yield from self.encode_chunk(s)

    def decode(self, ids: list[int]) -> str:
        if not ids:
            return ""
        if min(ids) < 0 or max(ids) >= len(self.vocab):
            raise ValueError("Invalid token id")
        b = b''.join(self.vocab[id] for id in ids)
        return b.decode("utf-8", errors="replace")

# Test
if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    tokenizer = BPE_tokenizer.from_files("data/BPE_result/owt_train.pkl", special_tokens)
    with open('data/TinyStories-sample.txt','r') as f:
        ids = list(tokenizer.encode_iterable(f))
        print(tokenizer.decode(ids))