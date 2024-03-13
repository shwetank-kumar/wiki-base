import torch
import torch.nn as nn
import torch.nn.functional as F
<<<<<<< HEAD

=======
>>>>>>> 3aa2028 (wiki-2)
##-------------------------
class Feedforward(nn.Module):
    def __init__(self, emb_dim, dropout):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, 4*emb_dim),
            nn.ReLU(),
            nn.Linear(4*emb_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        return self.ff(x)
    
    
class Head(nn.Module):
    def __init__(self, emb_dim, head_size, block_length, dropout):
        super().__init__()
        self.key = nn.Linear(emb_dim, head_size, bias=False)
        self.query = nn.Linear(emb_dim, head_size, bias=False)
        self.value = nn.Linear(emb_dim, head_size, bias=False)
        # Define a register buffer
        self.register_buffer('tril', torch.tril(torch.ones(block_length,block_length)))
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        ## Initialize the vector
        k = self.key(x) # B,T,head_size
        q = self.query(x) # B,T,head_size
        v = self.value(x) # B,T,head_size
        wei = k @ torch.transpose(q, 1, 2) * C**-0.5 # B,T,head_size * B,head_size,T -> B,T,T
        wei = torch.masked_fill(wei, self.tril[:T,:T] == 0, float('-Inf')) # Only selecting till the Tth column will be esp important during generation o/w will expect maxx_length columns at everytime step and throw an error
        wei = self.dropout(F.softmax(wei, dim=-1)) # B,T,T
        # print(k.shape, q.shape, v.shape, wei.shape)
        out = wei @ v #B,T,T * B,T,H -> B,T,H i.e. 32,16,16 * 32,16,8 -> 32,16,8
        return out
    
class MultiHeadAttentionModuleList(nn.Module):
    def __init__(self, head_size, num_heads, emb_dim, block_length, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(emb_dim, head_size, block_length, dropout) for i in range(num_heads)]) # B,T,head_size*num_heads
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        return x
    
class BlockScratch(nn.Module):
    def __init__(self, emb_dim, num_heads, block_length, dropout):
        super().__init__()
        self.head_size = emb_dim // num_heads
        self.sa_head = MultiHeadAttentionModuleList(self.head_size, num_heads, emb_dim, block_length,dropout)
        self.ff = Feedforward(emb_dim, dropout)
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)

    def forward(self, x, targets=None):
        sa_out = self.sa_head(self.ln1(x))
        x = x + sa_out
        x = x + self.ff(self.ln2(x))
        return x
    
    
<<<<<<< HEAD
class Xformer_Scratch(nn.Module):
    def __init__(self, emb_dim, vocab_size, num_heads, block_length, dropout):
=======
# class Xformer_Scratch(nn.Module):
#     def __init__(self, emb_dim, vocab_size, num_heads, block_length, dropout):
#         super().__init__()
#         self.token_embedding = nn.Embedding(vocab_size + 1, emb_dim)
#         self.pos_embedding = nn.Embedding(vocab_size + 1, emb_dim)
#         self.blocks = nn.Sequential(
#             BlockScratch(emb_dim, num_heads, block_length, dropout), 
#             BlockScratch(emb_dim, num_heads, block_length, dropout),
#             BlockScratch(emb_dim, num_heads, block_length, dropout),
#             nn.LayerNorm(emb_dim)
#         )
#         self.lm_head = nn.Linear(emb_dim, vocab_size)

#     def forward(self, x, targets=None):
#         tok_emb = self.token_embedding(x)
#         pos_emb = self.pos_embedding(x)
#         x = tok_emb + pos_emb # B, T, emb_dim
#         # x = self.sa_head(x) # B, T, head_size
#         # x =  self.ff(x)
#         x = self.blocks(x)
#         logits = self.lm_head(x) # B, T, vocab_size

#         # if we are given some desired targets also calculate the loss
#         loss = None
#         if targets is not None:
#             loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

#         return(logits,loss)
    
class Xformer_Scratch(nn.Module):
    def __init__(self, emb_dim, vocab_size, num_heads, num_layers, block_length, dropout):
>>>>>>> 3aa2028 (wiki-2)
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size + 1, emb_dim)
        self.pos_embedding = nn.Embedding(vocab_size + 1, emb_dim)
        self.blocks = nn.Sequential(
            BlockScratch(emb_dim, num_heads, block_length, dropout), 
            BlockScratch(emb_dim, num_heads, block_length, dropout),
            BlockScratch(emb_dim, num_heads, block_length, dropout),
            BlockScratch(emb_dim, num_heads, block_length, dropout), 
            BlockScratch(emb_dim, num_heads, block_length, dropout),
            BlockScratch(emb_dim, num_heads, block_length, dropout),
            nn.LayerNorm(emb_dim)
        )
        self.lm_head = nn.Linear(emb_dim, vocab_size)

    def forward(self, x, targets=None):
        tok_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding(x)
        x = tok_emb + pos_emb # B, T, emb_dim
        # x = self.sa_head(x) # B, T, head_size
        # x =  self.ff(x)
        x = self.blocks(x)
        logits = self.lm_head(x) # B, T, vocab_size

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return(logits,loss)