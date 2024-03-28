import os
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import pickle

"""
Prepare the data for feeding into training.
1. Fine tune a standard tokenizer
2. Tokenize the dataset
3. Store tuple dataset (raw_tokens_list for each split, tokenized dataset HF object) in .bin files
"""

## Dataset path
train_data_file = 'input.txt'
val_data_file = 'more.txt'

## Tokenizer path
tokenizer_path = "./shakespeare_char_tokenizer"
tokens_path = "./shakespeare_char_tokens.bin"

## Get data
# dataset = load_dataset("wikitext", name="wikitext-2-raw-v1")
# splits = dataset.keys()
with open(train_data_file,'r') as f:
    train_data_text = f.read()

with open(val_data_file,'r') as f:
    val_data_text = f.read()

dataset = {"train": train_data_text, "validation": val_data_text}
splits = dataset.keys()
vocab = sorted(set(dataset["train"] + dataset['validation']))
vocab_size = len(vocab)

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
    
def batch_tokenize(dataset, tokenizer, batch_size=1024):
    tokenized_dataset = {}
    for key, examples in dataset.items():
        # Initialize the list for tokenized examples under the current key
        tokenized_dataset[key] = []
        for i in range(0, len(examples), batch_size):
            # Create batches of examples
            batch = examples[i:i+batch_size]
            # Tokenize the batch and extend the list of tokenized examples
            tokenized_batch = tokenizer.encode(batch)
            tokenized_dataset[key].extend(tokenized_batch)
    return tokenized_dataset
    

if __name__ == '__main__':
    
    tokenizer = CharTokenizer(vocab)

    if not os.path.exists(tokenizer_path):
        tokenizer.save_pretrained(tokenizer_path)


    batch_size = 1024  # Define your batch size

    tokenized_dataset = batch_tokenize(dataset, tokenizer, batch_size)

    # # Save tokens and tokenized dataset object to a .bin file (raw_tokens:dict, tokens: Dataset)
    with open(tokens_path, "wb") as f:
        pickle.dump((vocab_size, tokenized_dataset), f)




## Function to finetune a tokenizer from based on the dataset 
## Train a new tokenizer using Wiki dataset from GPT2 tokenizer
# batch_size = 1000
# vocab_size = 5000
# num_proc = 6 ## num cpu cores // 2

# def batch_iterator(splits):
#     for split in splits:
#         for i in range(0, len(dataset[split]), batch_size):
#             yield dataset[split][i : i + batch_size]

# def tune_tokenizer():
#     gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
#     print(gpt2_tokenizer.is_fast)
#     tokenizer = gpt2_tokenizer.train_new_from_iterator(batch_iterator(splits), vocab_size=25000)
#     return tokenizer

# def batch_tokenize(dataset, tokenizer, batch_size=1024):
#     tokenized_dataset = {}
#     for key, examples in dataset.items():
#         # Initialize the list for tokenized examples under the current key
#         tokenized_dataset[key] = []
#         for i in range(0, len(examples), batch_size):
#             # Create batches of examples
#             batch = examples[i:i+batch_size]
#             # Tokenize the batch and extend the list of tokenized examples
#             tokenized_batch = tokenizer.encode(batch, add_special_tokens=True)
#             tokenized_dataset[key].extend(tokenized_batch)
#     return tokenized_dataset



