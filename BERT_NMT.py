import mxnet as mx
from mxnet.gluon.rnn.rnn_layer import LSTM


class CRNN_mxnet:
    def __init__(self, num_classes, dropout=0., rnn_hidden_size=100,
                 inference=True, img_width=560, num_label=20):
        '''
        same as conv-lite-lstm in CnOcr
        :param inference: boolean
            indicates evaluation without training
        '''
        # 560 => 140 - 1 = 139
        seq_len_cmpr_ratio = 4
        self.seq_len = img_width // seq_len_cmpr_ratio - 1
        self.dropout = dropout
        self.inference = inference
        self.num_classes = num_classes
        self.rnn_hidden_size = rnn_hidden_size

    def convRelu(self, idx, input_data, kernel_size, layer_size, padding_size,
                 batch_norm=True):
        layer = mx.symbol.Convolution(
            name='conv-%d' % idx,
            data=input_data,
            kernel=kernel_size,
            pad=padding_size,
            num_filter=layer_size,
        )
        if batch_norm:
            layer = mx.sym.BatchNorm(data=layer, name='batchnorm-%d' % idx)
        layer = mx.sym.LeakyReLU(data=layer, name='leakyrelu-%d' % idx)
        return layer

    def bottle_conv(self, idx, input_data, kernel_size, layer_size, padding_size,
                    batch_norm=True):
        bottle_channel = layer_size // 2
        layer = mx.symbol.Convolution(
            name='conv-%d-1-1x1' % idx,
            data=input_data,
            kernel=(1, 1),
            pad=(0, 0),
            num_filter=bottle_channel,
        )
        layer = mx.sym.LeakyReLU(data=layer, name='leakyrelu-%d-1' % idx)
        layer = mx.symbol.Convolution(
            name='conv-%d' % idx,
            data=layer,
            kernel=kernel_size,
            pad=padding_size,
            num_filter=bottle_channel,
        )
        layer = mx.sym.LeakyReLU(data=layer, name='leakyrelu-%d-2' % idx)
        layer = mx.symbol.Convolution(
            name='conv-%d-2-1x1' % idx,
            data=layer,
            kernel=(1, 1),
            pad=(0, 0),
            num_filter=layer_size,
        )
        if batch_norm:
            layer = mx.sym.BatchNorm(data=layer, name='batchnorm-%d' % idx)
        layer = mx.sym.LeakyReLU(data=layer, name='leakyrelu-%d' % idx)
        return layer

    def gen_network(self, data):
        kernel_size = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
        padding_size = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]
        layer_size = [min(32 * 2 ** (i + 1), 512) for i in range(len(kernel_size))]

        net = self.convRelu(
            0, data, kernel_size[0], layer_size[0], padding_size[0]
        )
        net = self.convRelu(
            1, net, kernel_size[1], layer_size[1], padding_size[1], True
        )
        net = mx.sym.Pooling(
            data=net, name='pool-0', pool_type='max', kernel=(2, 2), stride=(2, 2)
        )
        net = self.convRelu(
            2, net, kernel_size[2], layer_size[2], padding_size[2]
        )
        net = self.convRelu(
            3, net, kernel_size[3], layer_size[3], padding_size[3], True
        )
        x = net = mx.sym.Pooling(
            data=net, name='pool-1', pool_type='max', kernel=(2, 2), stride=(2, 2)
        )
        net = self.bottle_conv(4, net, kernel_size[4], layer_size[4], padding_size[4])
        net = self.bottle_conv(5, net, kernel_size[5], layer_size[5], padding_size[5], True) + x
        net = mx.symbol.Pooling(
            data=net, name='pool-2', pool_type='max', kernel=(2, 2), stride=(2, 1)
        )
        net = self.bottle_conv(6, net, (4, 1), layer_size[5], (0, 0))
        if self.dropout > 0.:
            net = mx.symbol.Dropout(data=net, p=self.dropout)

        # res: bz x emb_size x seq_len
        net = mx.symbol.squeeze(net, axis=2)
        net = mx.symbol.transpose(net, axes=(2, 0, 1))
        seq_model = LSTM(self.rnn_hidden_size, 2, bidirectional=True)
        hidden_concat = seq_model(net)
        return hidden_concat

    def get_network(self, data=None):
        # placeholder of input data
        self.data = mx.sym.Variable('data')
        # Note that the name of label is `label` instead of the \
        # default `softmax_label` in mxnet
        self.label = mx.sym.Variable('label')
        output = self.gen_network(self.data)
        # => (batch_size, seq_len=139, hidden_size=200)
        output = mx.symbol.transpose(output, axes=(1, 0, 2))
        # => (seq_len * batch_size, rnn_hidden_size)
        output_reshape = mx.symbol.reshape(output, shape=(-3, -2))
        # => ((seq_len * batch_size), num_classes)
        pred = mx.sym.FullyConnected(data=output_reshape,
                                     num_hidden=self.num_classes,
                                     name='pred_fc')
        if self.inference:
            return mx.sym.Group([output, mx.sym.softmax(data=pred, name='softmax')])
        else:
            # training with CTC loss
            # => (seq_len, batch_size, num_classes)
            pred_ctc = mx.sym.Reshape(data=pred, shape=(-4, self.seq_len, -1, 0))
            loss = mx.sym.contrib.ctc_loss(data=pred_ctc, label=self.label)
            ctc_loss = mx.sym.MakeLoss(loss)
            softmax_class = mx.symbol.SoftmaxActivation(data=pred)
            softmax_loss = mx.sym.MakeLoss(softmax_class)
            softmax_loss = mx.sym.BlockGrad(softmax_loss)
            return mx.sym.Group([softmax_loss, ctc_loss])

#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.onnx import operators
import math
from collections import defaultdict
from numpy.random import uniform
from transformers import BertTokenizer, AlbertForMaskedLM


# Helper funcs
INCREMENTAL_STATE_INSTANCE_ID = defaultdict(lambda: 0)


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def _get_full_incremental_state_key(module_instance, key):
    module_name = module_instance.__class__.__name__

    # assign a unique ID to each module instance, so that incremental state is
    # not shared across module instances
    if not hasattr(module_instance, '_fairseq_instance_id'):
        INCREMENTAL_STATE_INSTANCE_ID[module_name] += 1
        module_instance._fairseq_instance_id = INCREMENTAL_STATE_INSTANCE_ID[module_name]

    return '{}.{}.{}'.format(module_name, module_instance._fairseq_instance_id, key)


def softmax(x, dim, onnx_trace=False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        # noinspection PyTypeChecker
        return F.softmax(x, dim=dim, dtype=torch.float32)


def get_incremental_state(module, incremental_state, key):
    """Helper for getting incremental state for an nn.Module."""
    full_key = _get_full_incremental_state_key(module, key)
    if incremental_state is None or full_key not in incremental_state:
        return None
    return incremental_state[full_key]


def set_incremental_state(module, incremental_state, key, value):
    """Helper for setting incremental state for an nn.Module."""
    if incremental_state is not None:
        full_key = _get_full_incremental_state_key(module, key)
        incremental_state[full_key] = value


def make_positions(tensor, padding_idx, onnx_trace=False):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    mask = tensor.ne(padding_idx).long()
    return torch.cumsum(mask, dim=1) * mask + padding_idx

class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0., bias=True,
                 add_bias_kv=False, add_zero_attn=False, self_attention=False,
                 encoder_decoder_attention=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, 'Self-attention requires query, key and '                                                              'value to be of the same size'


        if self.qkv_same_dim:
            self.in_proj_weight = nn.Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        else:
            self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.vdim))
            self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))

        if bias:
            self.in_proj_bias = nn.Parameter(torch.Tensor(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)
            nn.init.xavier_uniform_(self.q_proj_weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False, attn_mask=None):
        """Input shape: Time x Batch x Channel

        Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.in_proj_k(key)
                v = self.in_proj_v(key)

        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)],
                    dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)
            saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(bsz, self.num_heads, -1, self.head_dim)

            self._set_input_buffer(incremental_state, saved_state)

        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.shape == torch.Size([]):
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)],
                    dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if self.onnx_trace:
                attn_weights = torch.where(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    torch.Tensor([float("-Inf")]),
                    attn_weights.float()
                ).type_as(attn_weights)
            else:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float('-inf'),
                )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace,
        ).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if (self.onnx_trace and attn.size(1) == 1):
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_q(self, query):
        if self.qkv_same_dim:
            return self._in_proj(query, end=self.embed_dim)
        else:
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[:self.embed_dim]
            return F.linear(query, self.q_proj_weight, bias)

    def in_proj_k(self, key):
        if self.qkv_same_dim:
            return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)
        else:
            weight = self.k_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[self.embed_dim:2 * self.embed_dim]
            return F.linear(key, weight, bias)

    def in_proj_v(self, value):
        if self.qkv_same_dim:
            return self._in_proj(value, start=2 * self.embed_dim)
        else:
            weight = self.v_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[2 * self.embed_dim:]
            return F.linear(value, weight, bias)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer[k] = input_buffer[k].index_select(0, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return get_incremental_state(
            self,
            incremental_state,
            'attn_state',
        ) or {}

    def _set_input_buffer(self, incremental_state, buffer):
        set_incremental_state(
            self,
            incremental_state,
            'attn_state',
            buffer,
        )


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    if not export and torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm
            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


# Modules
class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            padding_idx: int,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.onnx_trace = False

    def forward(self, input, incremental_state=None, positions=None):
        """Input is expected to be of size [bsz x seqlen]."""
        assert (
            (positions is None) or (self.padding_idx is None)
        ), "If positions is pre-computed then padding_idx should not be set."

        if positions is None:
            if incremental_state is not None:
                # positions is the same for every token when decoding a single step
                positions = input.data.new(1, 1).fill_(self.padding_idx + input.size(1))
            else:
                positions = make_positions(
                    input.data, self.padding_idx, onnx_trace=self.onnx_trace,
                )
        return super().forward(positions)

    def max_positions(self):
        """Maximum number of supported positions."""
        if self.padding_idx is not None:
            return self.num_embeddings - self.padding_idx - 1
        else:
            return self.num_embeddings


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.onnx_trace = False
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, incremental_state=None, timestep=None, **kwargs):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = torch.onnx.operators.shape_as_tensor(input)
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = (timestep.int() + 1).long() if timestep is not None else seq_len
            if self.onnx_trace:
                return self.weights[self.padding_idx + pos, :].unsqueeze(1).repeat(bsz, 1, 1)
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = make_positions(input, self.padding_idx, onnx_trace=self.onnx_trace)
        if self.onnx_trace:
            flat_embeddings = self.weights.detach().index_select(0, positions.view(-1))
            embedding_shape = torch.cat((bsz.view(1), seq_len.view(1), torch.LongTensor([-1])))
            embeddings = torch.onnx.operators.reshape_from_tensor_shape(flat_embeddings, embedding_shape)
            return embeddings
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number


def PositionalEmbedding(
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        learned: bool = False,
):
    if learned:
        # if padding_idx is specified then offset the embedding ids by
        # this index and adjust num_embeddings appropriately
        # TODO: The right place for this offset would be inside
        # LearnedPositionalEmbedding. Move this there for a cleaner implementation.
        if padding_idx is not None:
            num_embeddings = num_embeddings + padding_idx + 1
        m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        if padding_idx is not None:
            nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(
            embedding_dim, padding_idx, init_size=num_embeddings + padding_idx + 1,
        )
    return m


# Layers
class BERTfusedEncoderLayer(nn.Module):
    def __init__(self, embed_dim, encoder_ffn_embed_dim,
                 attention_dropout, dropout, bert_out_dim,
                 encoder_attention_heads,
                 encoder_ratio=0.5, bert_ratio=0.5,
                 bert_gate=True,
                 normalize_before=False,
                 bert_dropnet=False,
                 bert_dropnet_rate=0.25,
                 bert_mixup=False,
                 activation_dropout=0.,
                 **kwargs):
        """
        (bert_ratio, encoder_ratio) and dropnet are alternative
        """
        super(BERTfusedEncoderLayer, self).__init__()
        self.activation_fn = F.relu
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.normalize_before = normalize_before
        self.embed_dim = embed_dim
        self.self_attn = MultiheadAttention(
            embed_dim, encoder_attention_heads,
            dropout=attention_dropout, self_attention=True)
        self.bert_attn = MultiheadAttention(
            embed_dim=embed_dim, num_heads=encoder_attention_heads,
            kdim=bert_out_dim, vdim=bert_out_dim,
            dropout=attention_dropout,
        )
        self.self_attn_layer_norm = LayerNorm(embed_dim)
        self.fc1 = Linear(embed_dim, encoder_ffn_embed_dim)
        self.fc2 = Linear(encoder_ffn_embed_dim, embed_dim)
        self.final_layer_norm = LayerNorm(embed_dim)
        # bert-fused
        self.encoder_ratio = encoder_ratio
        self.bert_ratio = bert_ratio
        self.bert_dropnet = bert_dropnet
        self.bert_dropnet_rate = bert_dropnet_rate
        assert 0. <= self.bert_dropnet_rate <= 0.5
        self.bert_mixup = bert_mixup
        if not bert_gate:
            self.bert_ratio = 0.
            self.bert_dropnet = False
            self.bert_mixup = False

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {
            '0': 'self_attn_layer_norm',
            '1': 'final_layer_norm'
        }
        for old, new in layer_norm_map.items():
            for m in ('weight', 'bias'):
                k = '{}.layer_norms.{}.{}'.format(name, old, m)
                if k in state_dict:
                    state_dict[
                        '{}.{}.{}'.format(name, new, m)
                    ] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, bert_encoder_out,
                bert_encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        x1, _ = self.self_attn(
            query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x2, _ = self.bert_attn(
            query=x, key=bert_encoder_out, value=bert_encoder_out,
            key_padding_mask=bert_encoder_padding_mask)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        # DCMMC: drop-net trick
        ratios = self.get_ratio()
        x = residual + ratios[0] * x1 + ratios[1] * x2
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        return x

    def get_ratio(self):
        if self.bert_dropnet:
            frand = float(uniform(0, 1))
            if self.bert_mixup and self.training:
                return [frand, 1 - frand]
            # dropnet trick
            if frand < self.bert_dropnet_rate and self.training:
                return [1, 0]
            elif frand > 1 - self.bert_dropnet_rate and self.training:
                return [0, 1]
            else:
                return [0.5, 0.5]
        else:
            return [self.encoder_ratio, self.bert_ratio]

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x


class BERTfusedEncoder(nn.Module):
    def __init__(self, dropout, encoder_layer, embed_dim,
                 input_dim,
                 bert_out_dim, encoder_ffn_embed_dim,
                 encoder_attention_heads, attention_dropout,
                 encoder_normalize_before=False,
                 bert_dropnet=False,
                 bert_dropnet_rate=0.25,
                 bert_mixup=False,
                 **kwargs):
        super(BERTfusedEncoder, self).__init__()
        # what if add position embed?
        # self.embed_positions
        self.dropout = dropout
        self.bert_gates = [1, ] * encoder_layer
        self.layers = nn.ModuleList([])
        self.layers.extend([
            BERTfusedEncoderLayer(
                embed_dim=embed_dim,
                encoder_ffn_embed_dim=encoder_ffn_embed_dim,
                attention_dropout=attention_dropout,
                dropout=dropout,
                bert_out_dim=bert_out_dim,
                encoder_attention_heads=encoder_attention_heads,
                bert_gate=self.bert_gates[i],
                normalize_before=encoder_normalize_before,
                bert_dropnet=bert_dropnet,
                bert_dropnet_rate=bert_dropnet_rate,
                bert_mixup=bert_mixup,
            )
            for i in range(encoder_layer)
        ])
        self.project_in_dim = Linear(input_dim, embed_dim, bias=False) \
            if embed_dim != input_dim else None
        if encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def forward(self, source, src_lengths, encoder_padding_mask,
                bert_encoder_out):
        if self.project_in_dim is not None:
            source = self.project_in_dim(source)
        x = F.dropout(source, p=self.dropout, training=self.training)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        # encoder_padding_mask from CRNN
        for layer in self.layers:
            x = layer(
                x, encoder_padding_mask,
                bert_encoder_out['bert_encoder_out'],
                bert_encoder_out['bert_encoder_padding_mask'])
        if self.layer_norm:
            x = self.layer_norm(x)
        return {
            # T x B x C
            'encoder_out': x,
            # B x T
            'encoder_padding_mask': encoder_padding_mask
        }


class BERTfusedDecoderLayer(nn.Module):
    def __init__(self, embed_dim, decoder_ffn_embed_dim,
                 attention_dropout, dropout, bert_out_dim,
                 decoder_attention_heads,
                 encoder_ratio=0.5, bert_ratio=0.5,
                 normalize_before=False,
                 bert_dropnet=False,
                 bert_dropnet_rate=0.25,
                 bert_mixup=False,
                 no_encoder_attn=False, add_bias_kv=False,
                 add_zero_attn=False, bert_gate=True,
                 char_inputs=False,
                 activation_dropout=0.,
                ):
        super(BERTfusedDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=decoder_attention_heads,
            dropout=attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=True
        )
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.activation_fn = F.relu
        self.normalize_before = normalize_before
        self.embed_dim = embed_dim
        # dont know whats this
        export = char_inputs
        self.self_attn_layer_norm = LayerNorm(embed_dim, export=export)
        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = MultiheadAttention(
                embed_dim, decoder_attention_heads,
                dropout=attention_dropout, encoder_decoder_attention=True
            )
            self.bert_attn = MultiheadAttention(
                self.embed_dim, decoder_attention_heads,
                kdim=bert_out_dim, vdim=bert_out_dim,
                dropout=attention_dropout, encoder_decoder_attention=True
            )
            self.encoder_attn_layer_norm = LayerNorm(embed_dim, export=export)
        self.fc1 = Linear(self.embed_dim, decoder_ffn_embed_dim)
        self.fc2 = Linear(decoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True
        self.onnx_trace = False
        self.encoder_ratio = encoder_ratio
        self.bert_ratio = bert_ratio
        self.bert_dropnet = bert_dropnet
        self.bert_dropnet_rate = bert_dropnet_rate
        assert 0 <= self.bert_dropnet_rate <= 0.5
        self.bert_mixup = bert_mixup
        if not bert_gate:
            self.bert_ratio = 0.
            self.bert_dropnet = False
            self.bert_mixup = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(
        self,
        x,
        encoder_out=None,
        encoder_padding_mask=None,
        bert_encoder_out=None,
        bert_encoder_padding_mask=None,
        incremental_state=None,
        prev_self_attn_state=None,
        prev_attn_state=None,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``True``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        if self.encoder_attn is not None:
            residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
            if prev_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_attn_state
                saved_state = {"prev_key": prev_key, "prev_value": prev_value}
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)
            x1, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=(not self.training and self.need_attn),
            )
            x2, _ = self.bert_attn(
                query=x,
                key=bert_encoder_out,
                value=bert_encoder_out,
                key_padding_mask=bert_encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=(not self.training and self.need_attn),
            )
            x1 = F.dropout(x1, p=self.dropout, training=self.training)
            x2 = F.dropout(x2, p=self.dropout, training=self.training)
            ratios = self.get_ratio()
            x = residual + ratios[0] * x1 + ratios[1] * x2
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            self_attn_state = saved_state["prev_key"], saved_state["prev_value"]
            return x, attn, self_attn_state
        return x, attn

    def get_ratio(self):
        if self.bert_dropnet:
            frand = float(uniform(0, 1))
            if self.bert_mixup and self.training:
                return [frand, 1 - frand]
            if frand < self.bert_dropnet_rate and self.training:
                return [1, 0]
            elif frand > 1 - self.bert_dropnet_rate and self.training:
                return [0, 1]
            else:
                return [0.5, 0.5]
        else:
            return [self.encoder_ratio, self.bert_ratio]

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


class BERTfusedDecoder(nn.Module):
    def __init__(self,
                 num_tgt_alphabet,
                 dropout, decoder_layer, embed_dim,
                 bert_out_dim, decoder_ffn_embed_dim,
                 decoder_attention_heads, attention_dropout,
                 embed_layer,
                 normalize_before=False,
                 bert_dropnet=False,
                 bert_dropnet_rate=0.25,
                 bert_mixup=False,
                 max_target_positions=128,
                 no_token_positional_embeddings=False,
                 decoder_learned_pos=False,
                 decoder_no_bert=False,
                 no_encoder_attn=False):
        super(BERTfusedDecoder, self).__init__()
        bert_gates = [1, ] * decoder_layer
        self.layers = nn.ModuleList([])
        # TODO
        # if decoder_no_bert:
        self.layers.extend([
            BERTfusedDecoderLayer(
                embed_dim=embed_dim,
                decoder_ffn_embed_dim=decoder_ffn_embed_dim,
                attention_dropout=attention_dropout,
                dropout=dropout,
                bert_out_dim=bert_out_dim,
                decoder_attention_heads=decoder_attention_heads,
                normalize_before=normalize_before,
                bert_dropnet=bert_dropnet,
                bert_dropnet_rate=bert_dropnet_rate,
                bert_mixup=bert_mixup,
                bert_gate=bert_gates[i])
            for i in range(decoder_layer)
        ])
        self.dropout = dropout
        self.adaptive_softmax = None
        self.embed_layer = embed_layer
        self.project_in_dim = Linear(embed_layer.embedding_dim, embed_dim, bias=False) \
            if embed_dim != embed_layer.embedding_dim else None
        self.embed_scale = math.sqrt(embed_dim)
        out_embed_dim = self.embed_layer.embedding_dim
        padding_idx = self.embed_layer.padding_idx
        self.embed_positions = PositionalEmbedding(
            max_target_positions, embed_dim, padding_idx,
            learned=decoder_learned_pos,
        ) if not no_token_positional_embeddings else None
        self.project_out_dim = Linear(embed_dim, out_embed_dim, bias=False) \
            if embed_dim != out_embed_dim else None
        self.embed_out = nn.Parameter(torch.Tensor(num_tgt_alphabet, out_embed_dim))
        nn.init.normal_(self.embed_out, mean=0, std=out_embed_dim ** -0.5)
        if normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def forward(self, prev_output_tokens, encoder_out=None, bert_encoder_out=None,
                incremental_state=None, **unused):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(prev_output_tokens, encoder_out, bert_encoder_out,
                                         incremental_state)
        x = self.output_layer(x)
#         return x, extra
        return x

    def extract_features(self, prev_output_tokens, encoder_out=None, bert_encoder_out=None,
                         incremental_state=None, **unused):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
        # embed tokens and positions
        x = self.embed_scale * self.embed_layer(prev_output_tokens)
        if self.project_in_dim is not None:
            x = self.project_in_dim(x)
        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]
        # decoder layers
        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                bert_encoder_out['bert_encoder_out'],
                bert_encoder_out['bert_encoder_padding_mask'],
                incremental_state,
                self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
            )
            inner_states.append(x)
        if self.layer_norm:
            x = self.layer_norm(x)
        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        if self.project_out_dim is not None:
            x = self.project_out_dim(x)
        return x, {'attn': attn, 'inner_states': inner_states}

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or \
                self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(fill_with_neg_inf(self._future_mask.resize_(
                dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        return F.linear(features, self.embed_out)


class BERTfused(nn.Module):
    """
    Args:
        bert_model_name (str): default is 4-layer ALBERT
    """
    def __init__(self, num_tgt_alphabet,
                 input_dim,
                 bert_model_name='voidful/albert_chinese_tiny',
                 bert_output_layer=-1,
                 bert_dropnet_rate=0.5,
                 bert_dropnet=True,
                 bert_mixup=False,
                 dropout=0.3,
                 attention_dropout=0.,
                 normalize_before=False,
                 decoder_no_bert=False,
                 no_encoder_attn=False,
                 encoder_layer=6, decoder_layer=6,
                 encoder_embed_dim=512, encoder_ffn_embed_dim=1024,
                 encoder_attention_heads=4, decoder_attention_heads=4,
                 decoder_embed_dim=512, decoder_ffn_embed_dim=1024,
                 **kwargs):
        super(BERTfused, self).__init__()
        self.dropout = dropout
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_encoder = AlbertForMaskedLM.from_pretrained(
            bert_model_name,
            output_hidden_states=True,
            output_attentions=True)
        for param in self.bert_encoder.parameters():
            param.requires_grad = False
        self.bert_out_dim = self.bert_encoder.config.hidden_size
        self.bert_output_layer = bert_output_layer
        self.encoder = BERTfusedEncoder(
            input_dim=input_dim,
            dropout=self.dropout,
            encoder_layer=encoder_layer,
            embed_dim=encoder_embed_dim,
            bert_out_dim=self.bert_out_dim,
            encoder_ffn_embed_dim=encoder_ffn_embed_dim,
            attention_dropout=attention_dropout,
            encoder_attention_heads=encoder_attention_heads,
            encoder_normalize_before=normalize_before,
            bert_dropnet=bert_dropnet,
            bert_dropnet_rate=bert_dropnet_rate,
            bert_mixup=bert_mixup,
        )
        self.decoder_emb = self.build_embedding(num_tgt_alphabet, decoder_embed_dim)
        self.decoder = BERTfusedDecoder(
            num_tgt_alphabet=num_tgt_alphabet,
            embed_layer=self.decoder_emb,
            embed_dim=decoder_embed_dim,
            decoder_layer=decoder_layer,
            dropout=dropout,
            bert_out_dim=self.bert_out_dim,
            decoder_ffn_embed_dim=decoder_ffn_embed_dim,
            attention_dropout=attention_dropout,
            decoder_attention_heads=decoder_attention_heads,
            normalize_before=normalize_before,
            bert_dropnet=bert_dropnet,
            bert_dropnet_rate=bert_dropnet_rate,
            bert_mixup=bert_mixup,
            decoder_no_bert=decoder_no_bert,
            no_encoder_attn=no_encoder_attn,
        )

    @staticmethod
    def build_embedding(num_emb, embed_dim, padding_idx=0):
        emb = nn.Embedding(num_emb, embed_dim, padding_idx=padding_idx)
        nn.init.normal_(emb.weight, mean=0, std=embed_dim ** -0.5)
        nn.init.constant_(emb.weight[padding_idx], 0)
        return emb

    def forward(self, source, prev_output_tokens, bert_input,
                encoder_padding_mask=None, src_lengths=None,
                **kwargs):
        """
        Args:
            source (LongTensor): hidden feature outputed from RNN in CRNN
                `(batch, src_len, hidden_size)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            bert_input (list of str): output string of CRNN
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
        """
        bert_input = self.bert_tokenizer.batch_encode_plus(
            bert_input, pad_to_max_length=True, add_special_tokens=True)
        # gpu context?
        # In huggingface transformers, padding mask is 0 instead of 1
        bert_encoder_padding_mask = (torch.tensor(bert_input['attention_mask']) == 0)
        bert_encoder_out =  self.bert_encoder(
            torch.tensor(bert_input['input_ids']),
            attention_mask=bert_encoder_padding_mask)[-2]
        bert_encoder_out = bert_encoder_out[self.bert_output_layer]
        bert_encoder_out = {
            # => (T, B, C)
            'bert_encoder_out': bert_encoder_out.permute(1,0,2).contiguous(),
            'bert_encoder_padding_mask': bert_encoder_padding_mask
        }
        if type(src_lengths) == type(None):
            # default is no padding in source
            # TODO for now, we only support that source without padding
            src_lengths = torch.LongTensor([source.shape[1], ] * source.shape[0])
        # TODO encoder_padding_mask seems duplicated with src_lengths
        if type(encoder_padding_mask) == type(None):
            encoder_padding_mask = torch.zeros(source.shape[:2], dtype=torch.bool)
        encoder_out = self.encoder(
            source, src_lengths=src_lengths,
            encoder_padding_mask=encoder_padding_mask,
            bert_encoder_out=bert_encoder_out)
        decoder_out = self.decoder(
            prev_output_tokens, encoder_out=encoder_out,
            bert_encoder_out=bert_encoder_out, **kwargs)
        return decoder_out

# thinc>=8.0.0a0
# !pip install --user -U git+git://github.com/DCMMC/thinc.git
from thinc.api import MXNetWrapper, prefer_gpu
from thinc.api import Model, PyTorchWrapper
from thinc.api import xp2torch, torch2xp, Adam, MXNetShim
from thinc.util import xp2mxnet, mxnet2xp, convert_recursive, is_xp_array, is_mxnet_array
from thinc.types import ArgsKwargs
from thinc.loss import Loss
import mxnet as mx
import os
import numpy as np
import h5py
from tqdm.notebook import tqdm
from mxnet.gluon import SymbolBlock
import cupy
import torch.nn.functional as F
from minpy.context import set_context, cpu, gpu

gpus = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = gpus
is_gpu = len(gpus) > 0
set_context(gpu(0))

# is_gpu = prefer_gpu()
# print("GPU:", is_gpu)
context = [mx.context.gpu(i) for i in range(len(gpus))] if len(gpus) else \
    [mx.context.cpu()]
num_classes = 6426
crnn_instance = CRNN_mxnet(num_classes, inference=True)
rnn_output, pred = crnn_instance.get_network()
input_symbol = crnn_instance.data
prefix = '/data/xiaowentao/.cnocr/1.1.0/conv-lite-lstm/cnocr-v1.1.0-conv-lite-lstm'
epoch = 47
batch_size = 4
data_shape = [('data', (batch_size, 1, 32, 560))]
label_width = 20

# pred_fc = network.get_internals()['pred_fc_output']
# It seems that Thinc Shim only support mxnet Gluon.
# Therefore, we first wrap the Symbol network into Gluon Block.
network = SymbolBlock(outputs=[rnn_output, pred], inputs=input_symbol)
print('mx context before load:', mx.current_context())
# load the parameetrs!
network.collect_params().load('%s-%04d' % (prefix, epoch) + '.params',
                              ctx=context)
print('mx context after load:', mx.current_context())
# Yet another way!
# with open('/tmp/crnn.json', 'w') as f:
#     f.write(sym.tojson())
# network = SymbolBlock.imports('/tmp/crnn.json', ['data'],
#                               '%s-%04d' % (prefix, epoch) + '.params',
#                               ctx=context)
# network.hybridize(static_alloc=True, static_shape=True)

def hacked_MXNetWrapper():
    def MXNetWrapper(
        mxnet_model,
        convert_inputs=None,
        convert_outputs=None,
        model_class=Model,
        model_name="mxnet",
    ):
        if convert_inputs is None:
            convert_inputs = convert_mxnet_default_inputs
        if convert_outputs is None:
            convert_outputs = convert_mxnet_default_outputs
        return model_class(
            model_name,
            forward,
            attrs={"convert_inputs": convert_inputs, "convert_outputs": convert_outputs},
            shims=[MXNetShim(mxnet_model)],
        )

    def forward(model, X, is_train):
        convert_inputs = model.attrs["convert_inputs"]
        convert_outputs = model.attrs["convert_outputs"]
        Xmxnet, get_dX = convert_inputs(model, X, is_train)
        Ymxnet, mxnet_backprop = model.shims[0](Xmxnet, is_train)
        Y, get_dYmxnet = convert_outputs(model, (X, Ymxnet), is_train)
        def backprop(dY):
            dYmxnet = get_dYmxnet(dY)
            # hack! we only need d_rnn_output, dont need d_pred
            dYmxnet.args = tuple(tuple([dYmxnet.args[0][0][0]]))
            dXmxnet = mxnet_backprop(dYmxnet)
            dX = get_dX(dXmxnet)
            return dX
        return Y, backprop

    def convert_mxnet_default_inputs(
        model, X, is_train
    ):
        xp2mxnet_ = lambda x: xp2mxnet(x, requires_grad=is_train)
        converted = convert_recursive(is_xp_array, xp2mxnet_, X)
        if isinstance(converted, ArgsKwargs):
            def reverse_conversion(dXmxnet):
                return convert_recursive(is_mxnet_array, mxnet2xp, dXmxnet)
            return converted, reverse_conversion
        elif isinstance(converted, dict):
            def reverse_conversion(dXmxnet):
                dX = convert_recursive(is_mxnet_array, mxnet2xp, dXmxnet)
                return dX.kwargs
            return ArgsKwargs(args=tuple(), kwargs=converted), reverse_conversion
        elif isinstance(converted, (tuple, list)):
            def reverse_conversion(dXmxnet):
                dX = convert_recursive(is_mxnet_array, mxnet2xp, dXmxnet)
                return dX.args
            return ArgsKwargs(args=tuple(converted), kwargs={}), reverse_conversion
        else:
            def reverse_conversion(dXmxnet):
                dX = convert_recursive(is_mxnet_array, mxnet2xp, dXmxnet)
                return dX.args[0]
            return ArgsKwargs(args=(converted,), kwargs={}), reverse_conversion

    def convert_mxnet_default_outputs(model, X_Ymxnet, is_train):
        X, Ymxnet = X_Ymxnet
        Y = convert_recursive(is_mxnet_array, mxnet2xp, Ymxnet)
        def reverse_conversion(dY):
            dYmxnet = convert_recursive(is_xp_array, xp2mxnet, dY)
            return ArgsKwargs(args=((Ymxnet,),), kwargs={"head_grads": dYmxnet})
        return Y, reverse_conversion

    return MXNetWrapper

# MXNet doesn't provide a Softmax layer but a .softmax() operation/method for \
# prediction and it integrates an internal softmax during training. So to be able\
# to integrate it with the rest of the components, you combine it with a Softmax() \
# Thinc layer using the chain combinator.
wrapper_mxnet_crnn = hacked_MXNetWrapper()(network)


class NMTfusedCRNN(Model):
    def __init__(self, crnn, alphabet, input_dim=200, is_gpu=False):
        super(NMTfusedCRNN, self).__init__('NMTfusedCRNN', forward=self._forward_impl)
        self.crnn = crnn
        self.alphabet = alphabet
        # pad, bos, eos + alphabet
        num_tgt_alphabet = len(alphabet)
        nmt = BERTfused(num_tgt_alphabet=num_tgt_alphabet,
                        input_dim=input_dim)
        if is_gpu:
            nmt = nmt.cuda()
        self.nmt = PyTorchWrapper(nmt)
        self._layers = [self.crnn, self.nmt]

    @staticmethod
    def _forward_impl(model, inputs, is_train=True):
        img, prev_output_tokens = inputs
        prev_output_tokens = xp2torch(prev_output_tokens, requires_grad=False)
        if is_train:
            (rnn_output, pred), backprop_crnn = model.crnn.begin_update(img)
        else:
            rnn_output, pred = model.crnn.predict(img)
        batch_size = img.shape[0]
        Yh = cupy.asnumpy(pred)
        prob = np.reshape(Yh, (-1, batch_size, Yh.shape[1]))
        res_crnn = []
        for i in range(batch_size):
            lp = np.argmax(prob[:, i, :], axis=-1)
            res_crnn.append(''.join([
                model.alphabet[ele] for idx, ele in enumerate(
                    lp) if (lp[idx] and (idx == 0 or (lp[idx] != lp[idx - 1])))
            ]))
        if is_train:
            word_probs, backprop_nmt = model.nmt.begin_update(
                (rnn_output, prev_output_tokens, res_crnn))
        else:
            word_probs = model.nmt.predict(
                (rnn_output, prev_output_tokens, res_crnn))
        # normalized
        word_probs = torch2xp(F.log_softmax(xp2torch(word_probs), dim=-1))

        def finish_update(d_word_probs):
            d_source, d_prev_output_tokens, _ = backprop_nmt(d_word_probs)
            d_img = backprop_crnn(d_source)
            return (d_img, d_prev_output_tokens)

        return word_probs, finish_update


class LabelSmoothedCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, padding_idx=0):
        super(LabelSmoothedCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.padding_idx = padding_idx

    def forward(self, lprobs, target):
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = target.requires_grad_(False).view(-1, 1).to(dtype=torch.long)
        non_pad_mask = target.ne(self.padding_idx)
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
        eps_i = self.smoothing / lprobs.size(-1)
        loss = (1. - self.smoothing) * nll_loss + eps_i * smooth_loss
        return loss


class LabelSmoothedCrossEntropyLoss(Loss):
    def __init__(self, smoothing=0.1, padding_idx=0):
        super(LabelSmoothedCrossEntropyLoss, self).__init__()
        self.loss = PyTorchWrapper(
            LabelSmoothedCrossEntropy(smoothing, padding_idx))

    def get_loss(self, guesses, truths):
        return self.loss.predict((guesses, truths))

    def get_grad(self, guesses, truths):
        truths = truths.astype('float32')
        return self.loss.begin_update(
            (guesses, truths))[1]([torch.tensor(1.)])

    def __call__(self, guesses, truths):
        truths = truths.astype('float32')
        loss, grad_fn = self.loss.begin_update((guesses, truths))
        grad = grad_fn([torch.tensor(1.)])
        return grad, loss


imgs = []
golds = []
num_batch = 4
# source text (i.e. CRNN output) is not need to add bos and eos tokens,
# however, target text need.
alphabet = {(idx + 1): tok.strip() for idx, tok in enumerate(
    open('./cnocr/examples/label_cn.txt').readlines())}
# space token
alphabet[len(alphabet)] = ' '
pad_idx, bos_idx, eos_idx = 0, len(alphabet) + 1, len(alphabet) + 2
# the bos_idx and eos_idx is different from NMT where bos=2 and eos=3
alphabet[bos_idx] = '<s>'
alphabet[eos_idx] = '</s>'
# blank token
alphabet[0] = '#'
print(f'#alphabet:{len(alphabet)}, bos:{bos_idx}, eos:{eos_idx}')
alphabet_inv = {v: k for k, v in alphabet.items()}
mx_model = NMTfusedCRNN(wrapper_mxnet_crnn, alphabet, is_gpu=is_gpu)

with h5py.File('./data_generated/dataset_fonts.h5', 'r') as d:
    for idx in range(batch_size * num_batch):
        idx = str(idx)
        imgs.append(d[idx]['img'][...] / 255.)
        label = str(d[idx]['y'][...])
        golds.append(label)
imgs = np.expand_dims(np.array(imgs, dtype=np.float32), axis=1)
print('imgs:', imgs.shape)
imgs = mx_model.crnn.ops.asarray(imgs, dtype='float32')

golds_ids = [([alphabet_inv[c] for c in g] + [eos_idx, ]) for g in golds]
# In NMT, target usually uses right padding, but source text usually uses left padding.
golds_ids = [g + [pad_idx] * (label_width - len(g)) for g in golds_ids]
# used for Transformer's decoder
prev_output_tokens = [(g[-1:] + [bos_idx, ] + g[1:-1]) for g in golds_ids]
golds_ids = mx_model.crnn.ops.asarray(golds_ids, dtype='long')
prev_output_tokens = mx_model.crnn.ops.asarray(prev_output_tokens, dtype='long')

# mx_model.initialize(X=imgs[:4], Y=np.ones([4 * 139, num_classes], dtype=np.float32)[:4])
batches = mx_model.crnn.ops.multibatch(batch_size, imgs, prev_output_tokens,
                                       golds_ids, shuffle=True)


def inverse_sqrt_lr(lr=0.0005, warmup_updates=4000, warmup_init_lr=1e-7):
    num_updates = 0
    warmup_end_lr = lr
    lr_step = (warmup_end_lr - warmup_init_lr) / warmup_updates
    decay_factor = warmup_end_lr * warmup_updates ** 0.5
    while True:
        num_updates += 1
        if num_updates <= warmup_updates:
            yield warmup_init_lr + num_updates * lr_step
        else:
            yield decay_factor * num_updates ** -0.5


optimizer = Adam(
    beta1=0.9,
    beta2=0.98,
    eps=1e-10,
    learn_rate=inverse_sqrt_lr(),
    #     L2=1e-6,
    #     grad_clip=1.0,
    #     use_averages=True,
    #     L2_is_weight_decay=True
)
calculate_loss = LabelSmoothedCrossEntropyLoss()

res = []
ground = []
for X, tgt_shifted, Y in tqdm(batches, leave=True):
    # hidden: (batch_size, seq_len=139, hidden_size=200)
    #     hidden, Yh = mx_model.predict(X)
    Yh, backprop = mx_model.begin_update((X, tgt_shifted))
    print('Yh:', Yh.shape)
    grad, loss = calculate_loss(Yh, Y)
    print('loss:', loss)
    # grad = [d_guesses, d_truths]
    grad = grad[0]
    backprop(grad)
    print('mx context after load 2:', mx.current_context())
    mx_model.finish_update(optimizer)
    print('mx context after load 3:', mx.current_context())
    optimizer.step_schedules()
#     assert Yh.shape == (batch_size * 139, num_classes)
#     Yh = cupy.asnumpy(Yh)
#     Y = cupy.asnumpy(Y[0])
#     prob = np.reshape(Yh, (-1, batch_size, Yh.shape[1]))
#     for i in range(batch_size):
#         lp = np.argmax(prob[:, i, :], axis=-1)
# #         print(lp[:10])
#         res.append(''.join([
#             alphabet[ele] for idx, ele in enumerate(
#                 lp) if (lp[idx] and (idx == 0 or (lp[idx] != lp[idx - 1])))
#         ]))
#         ground.append(''.join([alphabet[c] for c in Y[i]]))
# print(res[:4], '\n', ground[:4])