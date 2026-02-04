from typing import DefaultDict
from collections import Counter, defaultdict
from sortedcontainers import SortedList
from llist import dllist, dllistnode

def update_count(
    pq: SortedList[tuple[int, tuple[bytes, bytes]]],
    byte_pair_count: Counter[tuple[bytes, bytes]],
    old_pair: tuple[bytes, bytes],
    new_pair: tuple[bytes, bytes],
    count: int,
):
    # Number of old_pair -= count, number of new_pair += count, and updating priority queue.
    pq.discard((byte_pair_count[old_pair], old_pair))
    pq.discard((byte_pair_count[new_pair], new_pair))
    byte_pair_count[old_pair] -= count
    byte_pair_count[new_pair] += count
    pq.add((byte_pair_count[old_pair], old_pair))
    pq.add((byte_pair_count[new_pair], new_pair))

def insert_ptr(
    ptr_set: set[tuple[int, dllistnode]],
    curr_opt_id: int,
    opt_id_dict: dict[dllistnode, int],
    node: dllistnode,
) -> int:
    ptr_set.add((curr_opt_id, node))
    opt_id_dict[node] = curr_opt_id
    return curr_opt_id + 1


def BPE_merge(
    pretoken_count: Counter[bytes],
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # Initialize the vocabulary with bytes 0-255 and special tokens
    vocab: list[bytes] = [token.encode("utf-8") for token in special_tokens]
    merges: list[tuple[bytes, bytes]] = []
    vocab.extend([bytes([i]) for i in range(256)])

    # Initialize the byte-pair counter and pointer
    byte_pair_count: Counter[tuple[bytes, bytes]] = Counter()
    byte_pair_ptr: DefaultDict[tuple[bytes, bytes], set[tuple[int, dllistnode]]] = defaultdict(set)
    # Record each doubly linked list (pretoken) and its count
    dl_count: Counter[int] = Counter()
    dl_store: list[dllist] = []
    # opt_id records the # of insert operations to byte_pair_ptr
    # We always iterate byte pairs from left to right, so same byte pairs at left will always have smaller opt_id
    # Hence, in byte_pair_ptr we will sort nodes by opt_id
    opt_id:int = 0
    opt_id_dict:dict[dllistnode, int] = {}
    
    for pretoken, count in pretoken_count.items():
        # Initialize and store the doubly linked list for the pretoken
        dl = dllist([pretoken[i:i+1] for i in range(len(pretoken))])
        dl_store.append(dl)
        dl_count[id(dl)] = count
        node = dl.first
        for i in range(len(pretoken) - 1):
            bp = (pretoken[i:i+1], pretoken[i+1:i+2])  # tuple of bytes
            byte_pair_count[bp] += count
            opt_id = insert_ptr(byte_pair_ptr[bp], opt_id, opt_id_dict, node)
            node = node.next
    
    # Initialize the priority queue, perform merge
    pq: SortedList[tuple[int, tuple[bytes, bytes]]] = SortedList(
        (count, pair) for pair, count in byte_pair_count.items()
    )
    while len(vocab) < vocab_size and pq:
        # Merge byte pair which appears most frequently
        count, best_pair = pq.pop(-1)
        if count == 0:
            break
        merged_bytes = best_pair[0] + best_pair[1]
        vocab.append(merged_bytes)
        merges.append(best_pair)

        # Update byte_pair_count and byte_pair_ptr
        for _, node in sorted(byte_pair_ptr[best_pair]):
            # May be lazy-deleted before as best_pair[0]
            dl = node.owner
            if dl is None:
                continue
            dl = dl()
            # May be lazy-deleted before as best_pair[1]
            if node.next.value != best_pair[1]:
                continue

            corr_pretoken_count = dl_count[id(dl)]
            newnode = dllistnode(merged_bytes)

            # Shape: prev bp[0](node) bp[1] next
            prev = node.prev
            if prev is not None:
                old_pair = (prev.value, node.value)
                new_pair = (prev.value, merged_bytes)
                update_count(pq, byte_pair_count, old_pair, new_pair, corr_pretoken_count)
                # Avoid modify byte_pair_ptr[best_pair] itself, it will be deleted later.
                if old_pair != best_pair:
                    byte_pair_ptr[old_pair].remove((opt_id_dict[prev], prev))
                opt_id = insert_ptr(byte_pair_ptr[new_pair], opt_id, opt_id_dict, prev)

            next = node.next.next
            if next is not None:
                old_pair = (node.next.value, next.value)
                new_pair = (merged_bytes, next.value)
                
                update_count(pq, byte_pair_count, old_pair, new_pair, corr_pretoken_count)
                # Avoid modify byte_pair_ptr[best_pair] itself, it will be deleted later.
                if old_pair != best_pair:
                    byte_pair_ptr[old_pair].remove((opt_id_dict[node.next], node.next))
                opt_id = insert_ptr(byte_pair_ptr[new_pair], opt_id, opt_id_dict, newnode)
            
            # Update doubly linked list
            dl.insertnode(newnode, node)
            dl.remove(node.next)
            dl.remove(node)

        pq.discard((byte_pair_count[best_pair], best_pair))
        byte_pair_count.pop(best_pair)
        byte_pair_ptr.pop(best_pair)
    
    return (dict(enumerate(vocab)), merges)


# Usage
if __name__ == "__main__":
    from pretokenization import pretokenization
    special_tokens = ["<|endoftext|>"]
    pretoken_count = pretokenization(
        filename="data/owt_train.txt",
        num_processes=14,
        special_tokens=special_tokens,
    )
    print(
        "Pretokens and sum of their lengths:",
        len(pretoken_count),
        sum(len(token) for token in pretoken_count.keys())
    )
    (vocab, merges) = BPE_merge(
        pretoken_count,
        vocab_size=32000,
        special_tokens=special_tokens
    )
    import pickle
    with open("data/BPE_result/TinyStoriesV2-GPT4-train.pkl", "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)
    