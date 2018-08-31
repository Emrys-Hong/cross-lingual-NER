import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('..')
from fastai.text import *

class LSTM_v1(object):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=False, only_use_last_hidden_state=False):
        """
        LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, length...).

        :param input_size:The number of expected features in the input x
        :param hidden_size:The number of features in the hidden state h
        :param num_layers:Number of recurrent layers.
        :param bias:If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first:If True, then the input and output tensors are provided as (batch, seq, feature)
        :param dropout:If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        :param bidirectional:If True, becomes a bidirectional RNN. Default: False
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias, 
                            batch_first=batch_first, dropout=dropout, 
                            bidirectional=bidirectional).cuda()
        
    def run(self, x, x_len):
        """
        sequence -> sort -> pad and pack ->process using RNN -> unpack ->unsort

        :param x: Variable
        :param x_len: numpy list
        :return:
        """
        """sort"""
        x_sort_idx = np.argsort(-x_len)
        x_unsort_idx = T(np.argsort(x_sort_idx))
        x_len = x_len[x_sort_idx]
        x = x[T(x_sort_idx)]
        """pack"""
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)
        """process using RNN"""
        out_pack, (ht, ct) = self.LSTM(x_emb_p,) # None)
        """unsort: h"""
        ht = torch.transpose(ht, 0, 1)[
            x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
        ht = torch.transpose(ht, 0, 1)

        if self.only_use_last_hidden_state:
            return ht
        else:
            """unpack: out"""
            out = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)  # (sequence, lengths)
            out = out[0]  #
            """unsort: out c"""
            out = out[x_unsort_idx]
            ct = torch.transpose(ct, 0, 1)[
                x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
            ct = torch.transpose(ct, 0, 1)

            return out, (ht, ct)

    def children(self):
        r"""Returns an iterator over immediate children modules.

        Yields:
            Module: a child module
        """
        for name, module in self.LSTM.named_children():
            yield module

    def parameters(self):
        r"""Returns an iterator over module parameters.

        This is typically passed to an optimizer.

        Yields:
            Parameter: module parameter

        Example::

            >>> for param in model.parameters():
            >>>     print(type(param.data), param.size())
            <class 'torch.FloatTensor'> (20L,)
            <class 'torch.FloatTensor'> (20L, 1L, 5L, 5L)

        """
        for name, param in self.LSTM.named_parameters():
            yield param


class LSTM_v2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=False, only_use_last_hidden_state=False):
        """
        LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, length...).

        :param input_size:The number of expected features in the input x
        :param hidden_size:The number of features in the hidden state h
        :param num_layers:Number of recurrent layers.
        :param bias:If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first:If True, then the input and output tensors are provided as (batch, seq, feature)
        :param dropout:If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        :param bidirectional:If True, becomes a bidirectional RNN. Default: False
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias,
                            batch_first=batch_first,# dropout=dropout, 
                            bidirectional=bidirectional).cuda()
    def children1(self):
        r"""Returns an iterator over immediate children modules.

        Yields:
            Module: a child module
        """
        for name, module in self.LSTM.named_children():
            yield module

    def parameters1(self):
        r"""Returns an iterator over module parameters.

        This is typically passed to an optimizer.

        Yields:
            Parameter: module parameter

        Example::

            >>> for param in model.parameters():
            >>>     print(type(param.data), param.size())
            <class 'torch.FloatTensor'> (20L,)
            <class 'torch.FloatTensor'> (20L, 1L, 5L, 5L)

        """
        for name, param in self.LSTM.named_parameters():
            yield param

    def forward(self, x, x_len):
        """
        sequence -> sort -> pad and pack ->process using RNN -> unpack ->unsort

        :param x: Variable
        :param x_len: numpy list
        :return:
        """
        """sort"""
        lengths_sorted, sorted_idx = x_len.sort(descending=True)
        x_sorted = x[sorted_idx]

        """pack"""
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x_sorted, lengths_sorted.tolist(), batch_first=self.batch_first)
        """process using RNN"""
        out_pack, (ht, ct) = self.LSTM(x_emb_p)

        if self.only_use_last_hidden_state:
            return ht
        else:
            """unpack: out"""
            h, _ = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first, padding_value=0)  # (sequence, lengths)
            h = torch.zeros_like(h).scatter_(0, sorted_idx.unsqueeze(1).unsqueeze(1).expand(-1, h.shape[1], h.shape[2]), h)
            ct = 1
            return h, (ht, ct)
