import os
from typing import BinaryIO
from multiprocessing import Pool
import regex as re
from collections import Counter
from functools import partial


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        
        # How much to backtrack to catch tokens spanning chunk boundaries
        overlap = len(split_special_token) - 1
        
        while True:
            mini_chunk = file.read(mini_chunk_size)

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            
            # Move forward, but backtrack by overlap to catch spanning tokens
            initial_position += mini_chunk_size - overlap
            file.seek(initial_position)

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def chunk_file(
    filename: str,
    num_processes: int,
    split_special_token: str,
) -> list[str]:
    chunks = []
    with open(filename, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_special_token.encode("utf-8"))
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)
    return chunks

def pretokenize_chunk(
    chunk: str,
    special_tokens: list[str],
) -> Counter[bytes]:
    # Split the chunk into parts to avoid special tokens
    pattern = "|".join(re.escape(t) for t in special_tokens)
    parts = re.split(pattern, chunk)
    
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    pretoken_count = Counter()
    for part in parts:
        for a in re.finditer(pattern, part):
            pretoken_count[a.group().encode("utf-8")] += 1
    
    return pretoken_count

def merge_counters(counters: list[Counter[bytes]]) -> Counter[bytes]:
    merged = Counter()
    for c in counters:
        merged.update(c)
    return merged

def pretokenization(
    filename: str,
    num_processes: int,
    special_tokens: list[str],
) -> Counter[bytes]:
    split_special_token = special_tokens[0]
    chunks = chunk_file(filename, num_processes, split_special_token)
    func = partial(pretokenize_chunk, special_tokens=special_tokens)

    with Pool(min(num_processes, len(chunks))) as pool:
        results = pool.map(func, chunks)
    return merge_counters(results)
    
# For test
if __name__ == "__main__":
    pretoken_count = pretokenization(
        filename="data/TinyStories-sample.txt",
        #filename="data/TinyStoriesV2-GPT4-valid.txt",
        num_processes=8,
        special_tokens=["<|endoftext|>"],
    )
    print(pretoken_count)