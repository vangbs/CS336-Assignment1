class BPE_tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        pass
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        pass
    def encode(self, text: str) -> list[int]:
        pass
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass
    def decode(self, ids: list[int]) -> str:
        pass
