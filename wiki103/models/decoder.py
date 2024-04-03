import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.dropout = dropout

    def forward(self, x):
        ## Using flash attention kernel
        ## Initialize the vector
        k = self.key(x) # B,T,head_size
        q = self.query(x) # B,T,head_size
        v = self.value(x) # B,T,head_size
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
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
    
class Xformer_Scratch(nn.Module):
    def __init__(self, emb_dim, vocab_size, num_heads, num_layers, block_length, dropout):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.pos_embedding = nn.Embedding(vocab_size, emb_dim)
        blocks = [BlockScratch(emb_dim, num_heads, block_length, dropout) for _ in range(num_layers)]
        blocks.append(nn.LayerNorm(emb_dim))
        
        self.blocks = nn.Sequential(*blocks)
        self.lm_head = nn.Linear(emb_dim, vocab_size)

    def forward(self, x, targets=None):
        tok_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding(x)
        x = tok_emb + pos_emb # B, T, emb_dim
        x = self.blocks(x)
        logits = self.lm_head(x) # B, T, vocab_size

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            # print(logits.view(-1, logits.size(-1)).shape, targets.view(-1).shape)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return(logits,loss)