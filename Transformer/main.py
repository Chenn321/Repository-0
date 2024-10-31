import torch
import torch.nn as nn
import math
from time import time

class MutiHeadAttention(nn.Module):
    def __init__(self, d_model, head, dropout=0.1):
        super().__init__()
        self.head = head
        self.d_model = d_model
        self.d_k = int(d_model // head)

        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)
        self.fc = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(dropout)
    
    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)   # transpose(-2,-1)交换最后两维
        if mask is not None:
            # mask = mask.unsqueeze(1)   # (batch, 1, seqLen) -> (batch, 1, 1, seqLen)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = torch.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, mask=None):
        batch = q.size(0)
        k = self.w_k(k).view(batch, -1, self.head, self.d_k)
        q = self.w_q(q).view(batch, -1, self.head, self.d_k)
        v = self.w_v(v).view(batch, -1, self.head, self.d_k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # (b, head, seqLen, dk)
        #print(q.shape)

        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
        #print(scores.shape)
        concat = scores.transpose(1, 2).contiguous().view(batch, -1, self.d_model)
        out = self.fc(concat)
        #print(out.shape)
        return out
    

class TransformerBlock(nn.Module):
    def __init__(self, d_model, head, dropout, forward_expansion):
        super().__init__()
        self.attention = MutiHeadAttention(d_model, head)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model*forward_expansion),
            nn.ReLU(),
            nn.Linear(d_model*forward_expansion, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask):
        attention = self.attention(q, k, v, mask)
        add = attention + q
        regular = self.dropout(self.layernorm1(add))
        forward = self.feed_forward(regular)
        out = self.dropout(self.layernorm2(forward + regular))
        return out



class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        max_len
    ):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding =  nn.Parameter(torch.zeros(1, max_len, d_model))  ###
        # self.position_embedding = nn.Embedding(max_len, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layers = nn.Sequential(
            *[
                TransformerBlock(
                    d_model,
                    heads,
                    dropout,
                    forward_expansion
                )
                for _ in range(num_layers)
            ]
        )
        
    def forward(self, x, mask):
        N, seq_length = x.shape

        word_embedding = self.word_embedding(x)
        position_embedding = self.position_embedding[:, :x.shape[1], :]
        # positions = torch.arange(0, seq_length).expand(N, seq_length)

        out = self.dropout(word_embedding + position_embedding)
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out



class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model,
        head,
        forward_expansion,
        dropout
    ):
        super().__init__()
        self.attention = MutiHeadAttention(d_model, head)
        self.transformer_block = TransformerBlock(
            d_model, 
            head, 
            dropout, 
            forward_expansion
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, src_mask, causal_mask):
        attention = self.attention(query, query, query, causal_mask)
        query = self.dropout(self.norm(attention + query))
        out = self.transformer_block(query, key, value, src_mask)
        return out

    def forward(self, x, k, v, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(query, k, v, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        num_layers,
        head,
        forward_expansion,
        dropout,
        max_len
    ):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        # self.position_embedding = nn.Embedding(max_len, d_model)
        self.layers = nn.Sequential(
            *[
            DecoderBlock(
                d_model,
                head,
                forward_expansion,
                dropout
            )
            for _ in range(num_layers)
        ]
        )
        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask, trg_mask):
        N, seq_length = x.shape
        # positions = torch.arange(0, seq_length).expand(N, seq_length)

        x = self.dropout(self.word_embedding(x) + self.positional_embedding[:, :x.shape[1], :])
        for layer in self.layers:
            x = layer(
                x, 
                encoder_output, 
                encoder_output, 
                src_mask, 
                trg_mask
            )
        out = self.fc(x)
        return out



class Transformer(nn.Module):
    def __init__(
        self,
        input_vocab_size,
        output_vocab_size,
        pad_idx,
        # trg_pad_idx,
        d_model=512,
        num_layers=6,
        forward_expansion=4,
        head=8,
        dropout=0,
        max_len=100
    ):
        super().__init__()
        self.encoder = Encoder(
            input_vocab_size,
            d_model,
            num_layers,
            head,
            forward_expansion,
            dropout,
            max_len
        )
        
        self.decoder = Decoder(
            output_vocab_size,
            d_model,
            num_layers,
            head,
            forward_expansion,
            dropout,
            max_len
        )
        
        self.pad_idx = pad_idx
        # self.trg_pad_idx = trg_pad_idx
    
    def pad_mask(self, inputs):
        pad_mask = (inputs != self.pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return pad_mask

    def trg_mask(self, target):
        N, target_len = target.shape
        target_mask = torch.tril(torch.ones((N, target_len, target_len))).unsqueeze(1)
        return target_mask

    def forward(self, inputs, target):
        pad_mask = self.pad_mask(inputs)
        trg_mask = self.trg_mask(target)
        encoder_output = self.encoder(inputs, pad_mask)
        decoder_out = self.decoder(target, encoder_output, pad_mask, trg_mask)
        return decoder_out
        

if __name__ == "__main__":

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]])

    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]])

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10

    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx)

    out = model(x, trg[:, :-1])
    print(out.shape)