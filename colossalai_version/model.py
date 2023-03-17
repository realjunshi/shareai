import math

import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import json

# n_embd = 512
n_embd = 448
# n_embd = 16
n_head = 8
# n_layer = 12
n_layer = 4

block_size = 400    # 一首歌最多1000个字
vocab_size = 7972
dropout = 0.2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)


class Head(nn.Module):
    """ 单头注意力机制"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)

        wei = q @ k.transpose(-2, -1) * (1 / math.sqrt(C))  # (B, T, C) @ (B, C, T) => (B, T, T)
        tril = torch.tril(torch.ones(T, T))
        tril = tril.to(device)
        wei = wei.masked_fill(tril == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        # 根据权重更新values
        v = self.value(x)  # (B, T, C)
        out = wei @ v  # (B, T, T) @ (B, T, C) => (B, T, C)

        return out


class MultiHeadAttention(nn.Module):
    """ 多头注意力机制 """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # print(x.shape)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # print(out.shape)
        out = self.dropout(self.proj(out))
        return out


# 直接将特征向量分成多份，进行多头训练
class MultiInnerHeadAttention(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)

        # 直接将特征向量划分成num_heads个头，每个头的长度是 C // num_heads
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, num_heads, T, C // num_heads)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, num_heads, T, C // num_heads)

        # (B, num_heads, T, C // num_heads) @ (B, num_heads, C // num_heads, T) => (B,num_heads, T, T)
        wei = q @ k.transpose(-2, -1) * (
                1 / math.sqrt(C))
        tril = torch.tril(torch.ones(T, T))
        tril = tril.to(device)
        wei = wei.masked_fill(tril == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        # 根据权重更新values
        v = self.value(x)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        # (B, num_heads, T, T) @ (B, num_heads, T, C // num_heads) => (B, num_heads,  T, C // num_heads)
        out = wei @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.proj(out)

        return out


class FeedForward(nn.Module):
    """前向网络"""

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        # 第一种多头注意力实现
        self.sa = MultiHeadAttention(n_head, head_size)
        # 第二种实现
        # self.sa = MultiInnerHeadAttention(n_head)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class LanguageModel(nn.Module):
    def __init__(self, vocab_size=vocab_size):
        super().__init__()
        self.tok_embedding = nn.Embedding(vocab_size, n_embd)
        # 加入位置编码
        self.pos_embedding = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            *(Block(n_embd, n_head=n_head) for _ in range(n_layer)),
            nn.LayerNorm(n_embd)
        )
        # 增加深度
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.tok_embedding(idx)
        pos_emb = self.pos_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb

        x = self.blocks(x)
        logits = self.lm_head(x)

        # if targets is None:
        #     loss = None
        # else:
        #     B, T, C = logits.shape
        #     logits = logits.view(B * T, C)
        #     # print(logits[0])
        #     targets = targets.view(B * T)
        #     # print(targets[0])
        #     loss = F.cross_entropy(logits, targets)

        return logits

    def generate(self, idx, max_new_token):
        # idx 开始字符的下标
        for _ in range(max_new_token):
            # 当长度超过每一个句子的长度时，取句子的后block_size长度
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            # 计算概率分布
            logits  = self(idx_cond)
            #  (B, C)
            logits = logits[:, -1, :]
            # softmax 获取概率分布 (B, C)
            probs = F.softmax(logits, dim=-1)
            # 取概率最大的下标作为下一个词 (B,1)
            id_next = torch.multinomial(probs, num_samples=1)
            # 将生成的词追加到已生成列表中
            idx = torch.cat((idx, id_next), dim=1)

        return idx

# 加载词表构建对应关系
def encode_decode():
    data_dir = os.path.join('./data/', "word_to_index")
    f = open(data_dir, 'r')
    data = json.load(f)

    stoi = data["stoi"]
    itos = data["itos"]

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[str(i)] for i in l])

    return encode, decode

def generate(model, begin, max_len=400):
    enc, dec = encode_decode()
    print(f"正在生成以 {begin} 开头的歌")
    begin_str = enc(begin)
    begin = torch.LongTensor(begin_str).to(device)
    begin = torch.unsqueeze(begin, 0)
    gen = model.generate(begin, max_new_token=max_len)[0].tolist()
    decode_song = dec(gen)
    return decode_song
