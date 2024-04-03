from transformers.tokenization_utils_base import PreTrainedTokenizerBase

class CharTokenizer(PreTrainedTokenizerBase):
    def __init__(self, vocab, **kwargs):
        super().__init__(vocab_files_names={},
            bos_token=None,
            eos_token=None,
            unk_token=None,
            sep_token=None,
            pad_token=None,
            cls_token=None,
            mask_token=None,
            **kwargs,)
        self.vocab = vocab
        self.stoi = {s: i+1 for i, s in enumerate(self.vocab)}
        self.itos = {i: s for s, i in self.stoi.items()}

    def encode(self, text_batch: list) -> list:
        encoded_text = [self.stoi[t] for t in text_batch]
        return encoded_text

    def decode(self, token_batch: list) -> list:
        decoded_text = ''.join([self.itos[t] for t in token_batch])
        return decoded_text
    
    def save_pretrained(self, save_directory: str):
        """
        Save the tokenizer vocabulary to a directory.
        """
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)

        vocab_file = os.path.join(save_directory, "vocab.txt")
        with open(vocab_file, "w", encoding="utf-8") as f:
            for token in self.vocab:
                f.write(token + "\n")

        return save_directory