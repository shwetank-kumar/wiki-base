import os
from datasets import load_dataset
from transformers import AutoTokenizer
from itertools import chain
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
tokenizer_path = "./shakespearetokenizer"
tokens_path = "./shakespearetokens.bin"

## Get data
# dataset = load_dataset("wikitext", name="wikitext-2-raw-v1")
# splits = dataset.keys()
with open(train_data_file,'r') as f:
    train_data_text = f.read()

with open(val_data_file,'r') as f:
    val_data_text = f.read()

dataset = {"train": train_data_text, "validation": val_data_text}
splits = dataset.keys()

## Function to finetune a tokenizer from based on the dataset 
## Train a new tokenizer using Wiki dataset from GPT2 tokenizer
batch_size = 1000
vocab_size = 5000
num_proc = 6 ## num cpu cores // 2

def batch_iterator(splits):
    for split in splits:
        for i in range(0, len(dataset[split]), batch_size):
            yield dataset[split][i : i + batch_size]

def tune_tokenizer():
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print(gpt2_tokenizer.is_fast)
    tokenizer = gpt2_tokenizer.train_new_from_iterator(batch_iterator(splits), vocab_size=25000)
    return tokenizer

def batch_tokenize(dataset, tokenizer, batch_size=1024):
    tokenized_dataset = {}
    for key, examples in dataset.items():
        # Initialize the list for tokenized examples under the current key
        tokenized_dataset[key] = []
        for i in range(0, len(examples), batch_size):
            # Create batches of examples
            batch = examples[i:i+batch_size]
            # Tokenize the batch and extend the list of tokenized examples
            tokenized_batch = tokenizer.encode(batch, add_special_tokens=True)
            tokenized_dataset[key].extend(tokenized_batch)
    return tokenized_dataset

if __name__ == '__main__':
    if not os.path.exists(tokenizer_path):
        tokenizer = tune_tokenizer()
        tokenizer.save_pretrained(tokenizer_path)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    batch_size = 1024  # Define your batch size

    tokenized_dataset = batch_tokenize(dataset, tokenizer, batch_size)

    # # Save tokens and tokenized dataset object to a .bin file (raw_tokens:dict, tokens: Dataset)
    with open(tokens_path, "wb") as f:
        pickle.dump((vocab_size, tokenized_dataset), f)