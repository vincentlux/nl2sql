import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

def run_lstm(lstm, input, input_len, hidden=None):
    # run the lstm using packed sequence
    # this requires to first sort the input according to its length
    # rank of each element
    sort_perm = np.array(sorted(range(len(input_len)),
        key=lambda k:input_len[k], reverse=True))
    # lengths
    # map rank to actual length sorted
    sort_input_len = input_len[sort_perm]
    # np.argsort: return the indices that would sort an array
    sort_perm_inv = np.argsort(sort_perm)

    if input.is_cuda:
        sort_perm = torch.LongTensor(sort_perm).cuda()
        sort_perm_inv = torch.LongTensor(sort_perm_inv).cuda()
    lstm_input = nn.utils.rnn.pack_padded_sequence(input[sort_perm],
     sort_input_len, batch_first=True)

    # ???
    if hidden is None:
        lstm_hidden = None
    else:
        # ???
        lstm_hidden = (hidden[0][:, sort_perm], hidden[1][:, sort_perm])
    
    # do lstm with param input
    sort_ret_s, sort_ret_h = lstm(lstm_input, lstm_hidden)
    # ???
    ret_s = nn.utils.rnn.pad_packed_sequence(sort_ret_s, batch_first=True)[0][sort_perm_inv]
    # ???
    ret_h = (sort_ret_h[0][:, sort_perm_inv], sort_ret_h[1][:, sort_perm_inv])
    return ret_s, ret_h


    # https://pytorch.org/docs/stable/nn.html?highlight=pack#torch.nn.utils.rnn.pack_padded_sequence
    # https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch

def col_name_encode(name_input_var, name_len, col_len, enc_lstm):
    # read papers
    # column attention
    # encode the columns
    # the embedding of a column name is the last state of its lstm output
    name_hidden, _ = run_lstm(enc_lstm, name_input_var, name_len)
    name_out = name_hidden[tuple(range(len(name_len))), name_len-1]
    ret = torch.FloatTensor(
            len(col_len), max(col_len), name_out.size()[1]).zero_()
    if name_out.is_cuda:
        ret = ret.cuda()

    st = 0
    for idx, cur_len in enumerate(col_len):
        ret[idx, :cur_len] = name_out.data[st:st+cur_len]
        st += cur_len
    ret_var = Variable(ret)
    return ret_var, col_len