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
    lstm_input = nn.utils.rnn.pack_padded_sequence(input=input[sort_perm],
     lengths=sort_input_len, batch_first=True)

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
        return 0