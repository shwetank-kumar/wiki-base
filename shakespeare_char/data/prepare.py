import os
import pickle
from chartokenizer import CharTokenizer

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

    # print(tokenizer.vocab)

    with open(os.path.join(tokenizer_path, 'vocab.txt'), "wb") as f:
        pickle.dump((tokenizer.vocab), f)