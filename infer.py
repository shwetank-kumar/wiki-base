from models.wiki2 import Xformer_Scratch as Xformer
from infer_config import * 
import pickle
import os
import torch
from accelerate import Accelerator
from transformers import AutoTokenizer
from torch.nn import functional as F

with open(dataset_file, "rb") as f:
            loaded_objects = pickle.load(f)

# Unpack the tuple of objects
vocab_size, tokenized_text, _ = loaded_objects

model = Xformer(emb_dim, vocab_size, num_heads, num_layers, block_size, dropout)
# print(model)
accelerator = Accelerator()
if os.path.join(checkpoint_dir, model_weight_file):
    # Load model
    model = Xformer(emb_dim, vocab_size, num_heads, num_layers, block_size, dropout)
    # Load model state dict
    modelwgts = torch.load(os.path.join(checkpoint_dir, model_weight_file), map_location=torch.device(accelerator.device.type))
    print(f"Loading checkpoint from {checkpoint_dir}")
    model.load_state_dict(modelwgts)
    model.to(accelerator.device.type)

else:
    print("Model checkpoint does not exist.")

if os.listdir(tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

else:
     print("Tokenizer model does not exist.")

@torch.no_grad()
def generate(model, idx, max_new_tokens, block_size=16):
    """Generates text given a model and a seed"""
    for _ in range(max_new_tokens):
        # print('idx shape:',idx.shape)
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        idx_cond = idx_cond
        logits, _ = model(idx_cond)
        # Pick only the logits from most recent time step. Karpathy also does a divide by temp?
        # This is just Platt scaling which makes the various Softmax curves closes adding more randomness
        # see scratch.ipynb. https://en.wikipedia.org/wiki/Platt_scaling
        logits = logits[:,-1,:]
        probs = F.softmax(logits, dim=-1)
        # print('prob dist:',probs)
        idx_next = torch.multinomial(probs, num_samples=1)
        # print('idx_next shape:',idx_next.shape)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx.squeeze().tolist()
    
seed = "==British Raj==="

seed_tokens = tokenizer.encode(seed)#["input_ids"]
decoded_seed = tokenizer.decode(seed_tokens)
# print(seed_tokens, decoded_seed)
seed_tensor = torch.unsqueeze(torch.tensor(seed_tokens, dtype=torch.long), dim=0).to(accelerator.device.type)
generated_tokens = generate(model, seed_tensor, block_size, block_size)
# print(generated_tokens)
print(tokenizer.decode(generated_tokens))