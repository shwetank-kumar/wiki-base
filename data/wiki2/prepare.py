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

## Tokenizer path
tokenizer_path = "./wiki2tokenizer"
tokens_path = "./wiki2tokens.bin"

## Get data
dataset = load_dataset("wikitext", name="wikitext-2-raw-v1")
splits = dataset.keys()


## Function to finetune a tokenizer from based on the dataset 
## Train a new tokenizer using Wiki dataset from GPT2 tokenizer
batch_size = 1000
vocab_size = 5000
num_proc = 6 ## num cpu cores // 2

def batch_iterator(splits):
    for split in splits:
        for i in range(0, len(dataset[split]), batch_size):
            yield dataset[split][i : i + batch_size]["text"]

def tune_tokenizer():
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print(gpt2_tokenizer.is_fast)
    tokenizer = gpt2_tokenizer.train_new_from_iterator(batch_iterator(splits), vocab_size=25000)
    return tokenizer

def tokenize(example):
    tokenized_dataset = tokenizer(example)
    return tokenized_dataset

if __name__ == '__main__':
    if not os.path.exists(tokenizer_path):
        tokenizer = tune_tokenizer()
        tokenizer.save_pretrained(tokenizer_path)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    tokenized_dataset = dataset.map(lambda example: tokenize(example['text']), 
                          batched=True,
                          desc="Tokenizing the splits",
                          num_proc=num_proc,)
    tokens = {}
    for s in tokenized_dataset:
        tokens[s] = list(chain.from_iterable(tokenized_dataset[s]['input_ids']))


    # Save tokens and tokenized dataset object to a .bin file (raw_tokens:dict, tokens: Dataset)
    with open(tokens_path, "wb") as f:
        pickle.dump((vocab_size, tokens), f)