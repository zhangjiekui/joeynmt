# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
from torch import Tensor


# pylint: disable=arguments-differ
'''
class MultiHeadedAttention_Origin(nn.Module):
    """
    Multi-Head Attention module from "Attention is All You Need"

    Implementation modified from OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(self, num_heads: int, size: int, dropout: float = 0.1):
        """
        Create a multi-headed attention layer.
        :param num_heads: the number of heads
        :param size: model size (must be divisible by num_heads)
        :param dropout: probability of dropping a unit
        """
        super().__init__()

        assert size % num_heads == 0

        self.head_size = head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads

        self.k_layer = nn.Linear(size, num_heads * head_size)
        self.v_layer = nn.Linear(size, num_heads * head_size)
        self.q_layer = nn.Linear(size, num_heads * head_size)

        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, k: Tensor, v: Tensor, q: Tensor, mask: Tensor = None):
        """
        Computes multi-headed attention.

        :param k: keys   [B, M, D] with M being the sentence length.
        :param v: values [B, M, D]
        :param q: query  [B, M, D]
        :param mask: optional mask [B, 1, M]
        :return:
        """
        batch_size = k.size(0)
        num_heads = self.num_heads

        # project the queries (q), keys (k), and values (v)
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)

        # reshape q, k, v for our computation to [batch_size, num_heads, ..]
        k = k.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)

        # compute scores
        q = q / math.sqrt(self.head_size)

        # batch x num_heads x query_len x key_len
        scores = torch.matmul(q, k.transpose(2, 3))

        # apply the mask (if we have one)
        # we add a dimension for the heads to it below: [B, 1, 1, M]
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))

        # apply attention dropout and compute context vectors.
        attention = self.softmax(scores)
        attention = self.dropout(attention)

        # get context vector (select values with attention) and reshape
        # back to [B, M, D]
        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, num_heads * self.head_size)

        output = self.output_layer(context)

        return output
'''

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads: int = 8, size: int = 256, dropout: float = 0.1):
        """
            Create a multi-headed attention layer.
            :param num_heads: the number of heads
            :param size: model size(embed_dim), must be divisible by num_heads
            :param dropout: probability of dropping a unit
        """

        super().__init__()
        assert size % num_heads == 0, f"size:int={size}必须能被num_heads={num_heads}整除！"
        self.head_size = head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads
        # print(num_heads * head_size==size)  #True

        # query、key、value
        self.k_layer = nn.Linear(size, size)
        self.v_layer = nn.Linear(size, size)
        self.q_layer = nn.Linear(size, size)

        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, k: Tensor, v: Tensor, q: Tensor, mask: Tensor = None):
        """
        Computes multi-headed attention.

        :param k: keys   [B, L, D] with L being the sentence length.
        :param v: values [B, L, D]
        :param q: query  [B, L, D]
        :param mask: optional mask [B, 1, L], element must by True or False
        :return:
        """
        batch_size = k.size(0)  # B: batch_size
        num_heads = self.num_heads

        # project the queries (q), keys (k), and values (v)
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)

        # reshape q, k, v for our computation to [batch_size, num_heads, ..]

        # result shape: [B:batch_size, num_heads, L:seq_len, head_size]
        k = k.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)

        # compute scores
        q = q / math.sqrt(self.head_size)

        # result shape: batch x num_heads x query_len x key_len
        scores = torch.matmul(q, k.transpose(2, 3))

        # apply the mask (if we have one)
        # we add a dimension for the heads to it below: [B, 1, 1, L]
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))

        # apply attention dropout and compute context vectors.
        attention = self.softmax(scores)  # seq_len中mask为True的位置的attention将为0.0
        attention_origin_returned=attention[:,0,:,:]
        attention = self.dropout(attention)

        # get context vector (select values with attention) and reshape
        # back to [B, L, D]
        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, num_heads * self.head_size)

        output = self.output_layer(context)

        return output, attention , attention_origin_returned


# pylint: disable=arguments-differ
class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-forward layer
    Projects to ff_size and then back down to input_size.
    """

    def __init__(self, input_size, ff_size, dropout=0.1):
        """
        Initializes position-wise feed-forward layer.
        :param input_size: dimensionality of the input.
        :param ff_size: dimensionality of intermediate representation
        :param dropout:
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.pwff_layer = nn.Sequential(
            nn.Linear(input_size, ff_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, input_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.layer_norm(x)
        return self.pwff_layer(x_norm) + x


# pylint: disable=arguments-differ
class PositionalEncoding(nn.Module):
    """
    Pre-compute position encodings (PE).
    In forward pass, this adds the position-encodings to the
    input for as many time steps as necessary.

    Implementation based on OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """
    def __init__(self,
                 size: int = 0,
                 max_len: int = 5000):
        """
        Positional Encoding with maximum length max_len
        :param size:
        :param max_len:
        :param dropout:
        """
        if size % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(size))
        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, size, 2, dtype=torch.float) *
                              -(math.log(10000.0) / size)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, size, max_len]
        super().__init__()
        self.register_buffer('pe', pe)
        self.dim = size

    def forward(self, emb):
        """Embed inputs.
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
        """
        # Add position encodings
        return emb + self.pe[:, :emb.size(1)]

# https://github.com/pytorch/pytorch/blob/68d438c9dade66073b3f9657bc077623c22001b9/torch/nn/modules/transformer.py#L241
# pytorch class TransformerEncoderLayer(Module)
class TransformerEncoderLayer(nn.Module):
    """
    One Transformer encoder layer has a Multi-head attention layer plus
    a position-wise feed-forward layer.
    """

    def __init__(self,
                 size: int = 0,
                 ff_size: int = 0,
                 num_heads: int = 0,
                 dropout: float = 0.1):
        """
        A single Transformer layer.
        :param size:
        :param ff_size:
        :param num_heads:
        :param dropout:
        """
        super().__init__()

        self.layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.src_src_att = MultiHeadedAttention(num_heads, size,
                                                dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(size, ff_size=ff_size,
                                                    dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.size = size

    # pylint: disable=arguments-differ
    # https://github.com/pytorch/pytorch/blob/68d438c9dade66073b3f9657bc077623c22001b9/torch/nn/modules/transformer.py#L282
    # def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Forward pass for a single transformer encoder layer.
        First applies layer norm, then self attention,
        then dropout with residual connection (adding the input to the result),
        and then a position-wise feed-forward layer.

        :param x: layer input
        :param mask: input mask
        :return: output tensor
        """
        x_norm = self.layer_norm(x)
        h = self.src_src_att(x_norm, x_norm, x_norm, mask)
        h = self.dropout(h) + x
        o = self.feed_forward(h)
        return o

# https://github.com/pytorch/pytorch/blob/68d438c9dade66073b3f9657bc077623c22001b9/torch/nn/modules/transformer.py#L303
# class TransformerDecoderLayer(Module):
class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer.

    Consists of self-attention, source-attention, and feed-forward.
    """

    def __init__(self,
                 size: int = 0,
                 ff_size: int = 0,
                 num_heads: int = 0,
                 dropout: float = 0.1):
        """
        Represents a single Transformer decoder layer.

        It attends to the source representation and the previous decoder states.

        :param size: model dimensionality
        :param ff_size: size of the feed-forward intermediate layer
        :param num_heads: number of heads
        :param dropout: dropout to apply to input
        """
        super().__init__()
        self.size = size

        self.trg_trg_att = MultiHeadedAttention(num_heads, size,
                                                dropout=dropout)
        self.src_trg_att = MultiHeadedAttention(num_heads, size,
                                                dropout=dropout)

        self.feed_forward = PositionwiseFeedForward(size, ff_size=ff_size,
                                                    dropout=dropout)

        self.x_layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.dec_layer_norm = nn.LayerNorm(size, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

    # pylint: disable=arguments-differ
    # https://github.com/pytorch/pytorch/blob/68d438c9dade66073b3f9657bc077623c22001b9/torch/nn/modules/transformer.py#L348
    # def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
    def forward(self,
                x: Tensor = None,
                memory: Tensor = None,
                src_mask: Tensor = None,
                trg_mask: Tensor = None) -> Tensor:
        """
        Forward pass of a single Transformer decoder layer.

        :param x: inputs
        :param memory: source representations
        :param src_mask: source mask
        :param trg_mask: target mask (so as to not condition on future steps)
        :return: output tensor
        """
        # decoder/target self-attention
        x_norm = self.x_layer_norm(x)
        h1 = self.trg_trg_att(x_norm, x_norm, x_norm, mask=trg_mask)
        h1 = self.dropout(h1) + x

        # source-target attention
        h1_norm = self.dec_layer_norm(h1)
        h2 = self.src_trg_att(memory, memory, h1_norm, mask=src_mask)

        # final position-wise feed-forward layer
        o = self.feed_forward(self.dropout(h2) + h1)

        return o

# 测试用例与用法示范
def MultiHeadedAttention_test():
    from torch.nn import MultiheadAttention as TorchMultiheadAttention
    global output
    mha = MultiHeadedAttention()
    mha_t=TorchMultiheadAttention(embed_dim=mha.model_size,num_heads=mha.num_heads,dropout=0.)
    print("mha  :",mha.head_size, mha.num_heads, mha.model_size)
    print("mha_t:", mha_t.head_dim, mha_t.num_heads, mha_t.embed_dim)

    B, L, D = 2, 5, mha.model_size  # shape 参数
    q = torch.randn(B, L, D)
    k = torch.ones(B, L, D)
    v = torch.rand(B, L, D)
    print(f"shape 参数, q:{q.shape} k:{k.shape} v:{v.shape}")
    # mask_int = torch.randint(0, 2, (B, 1, L))
    mask_int = torch.zeros((B, 1, L))
    # mask_int = torch.ones((B, 1, L))
    mask_bool = (mask_int == 1)

    q_t=q.permute(1,0,2)
    k_t=k.permute(1,0,2)
    v_t=v.permute(1,0,2)
    mask_bool_t=mask_bool.reshape(B,-1)
    print(f"mask_int:{str(mask_int)} ||\nmask_bool:{str(mask_bool)} ||\n")
    output = mha(q=q, k=k, v=v, mask=mask_bool)
    output_t = mha_t(query=q_t, key=k_t, value=v_t, key_padding_mask=mask_bool_t)
    print(f"output[0]   shape:{output[0].shape}") # output
    print(f"output[1]   shape:{output[1].shape}") # attn after dropout
    print(f"output[2]   shape:{output[2].shape}")  # attn without dropout, reshaped same to torch.nn.MultiheadAttention

    print(f"output_t[0] shape:{output_t[0].shape}")  # attn_output: (L:tgt len, N:batch, E:emb_dim)
    print(f"output_t[1] shape:{output_t[1].shape}")  # attn_output_weights: (N, L, S)(N,L,S:src len)

    print("***"*30)

    B, D = 2, mha.model_size  # shape 参数
    Lq=5
    Lkv=15
    q = torch.randn(B, Lq, D)
    k = torch.ones(B, Lkv, D)
    v = torch.rand(B, Lkv, D)

    print(f"shape 参数, q:{q.shape} k:{k.shape} v:{v.shape}")
    mask_int = torch.randint(0, 2, (B, 1, Lkv))
    mask_bool = (mask_int == True)
    print(f"mask_int:{str(mask_int)} ||\nmask_bool:{str(mask_bool)} ||\n")


    q_t=q.permute(1,0,2)
    k_t=k.permute(1,0,2)
    v_t=v.permute(1,0,2)
    mask_bool_t=mask_bool.reshape(B,-1)
    print(f"mask_int:{str(mask_int)} ||\nmask_bool:{str(mask_bool)} ||\n")
    output = mha(q=q, k=k, v=v, mask=mask_bool)
    output_t = mha_t(query=q_t, key=k_t, value=v_t, key_padding_mask=mask_bool_t)
    print(f"output[0]   shape:{output[0].shape}") # output
    print(f"output[1]   shape:{output[1].shape}") # attn after dropout
    print(f"output[2]   shape:{output[2].shape}")  # attn without dropout, reshaped same to torch.nn.MultiheadAttention

    print(f"output_t[0] shape:{output_t[0].shape}")  # attn_output: (L:tgt len, N:batch, E:emb_dim)
    print(f"output_t[1] shape:{output_t[1].shape}")  # attn_output_weights: (N, L, S)(N,L,S:src len)

    print("attention---"*10)

    print(f"output[2]   attention:\n{output[2][:,0,:].detach().numpy()}")
    print(f"output_t[1] attention:\n{output_t[1][:,0,:].detach().numpy()}")
    assert output[2].equal(output_t[1]),"attenstion 应该相等"


if __name__ == '__main__':
    MultiHeadedAttention_test()



