import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class WordEmbedding(nn.Module):
    '''
    Create word embeddings for questions and column names
    '''
    def __init__(self, word_emb, N_word, gpu, SQL_TOK,
     trainable=False):
        super(WordEmbedding, self).__init__()
        self.trainable = trainable
        self.N_word = N_word
        self.gpu = gpu
        self.SQL_TOK = SQL_TOK

        if trainable:
            print("Using trainable embedding")
            print("Did not implement trainable")
        else:
            '''
            word_emb directly from glove, eg.
            the 1.43 1.33...(dim)
            down -0.83 1.54...(dim)
            ...
            '''
            print("Using fixed embedding")
            self.word_emb = word_emb
            
    def gen_x_batch(self, questions, cols):
        '''
        generate word embeddings by batch size defined in train.py
        input: 
            questions: n (btach size) question arrays
                [[q1],[q2],..n]
            cols: column name of n (btach size) tables  arrays
                [[[t1col1],[t1col2]],[[t2col1],[t2col2]],..n]
                no use if only SQLNet
        output: 
            val_input_var: torch word_emb array with size(num_of_q * max_len * word_dim)
            val_lens: 1d array with length the batch size, saving the length of each question
        '''
        B = len(questions)
        val_embs = []
        # val_lens: length of the question
        val_lens = np.zeros(B, dtype=np.int64)
        # one_q: 1D tokenized question array
        # one_col: 2D tokenized col name array
        for i, (one_q, one_col) in enumerate(zip(questions, cols)):
            # get word embeddings of every single token of one_question and save together; if null, fill it with zeros
            # print("one_q", one_q)
            # print(self.word_emb.get('yes'))
            # print(self.word_emb.get(one_q[0], np.zeros(self.N_word, dtype=np.float32)))
            q_value = list(map(lambda x:self.word_emb.get(x, np.zeros(self.N_word, dtype=np.float32)), one_q))
            # for i in range(len(q_value)):
            # for a in range(len(q_value)):
            #     print("q_value[a]",np.array(list(q_value[a]), dtype='float32'))
            #     print("list_q_value[a]", list(q_value[a]))
            # add <BEG>and<END>
            val_embs.append([np.zeros(self.N_word, dtype=np.float32)] + q_value + [np.zeros(self.N_word, dtype=np.float32)])
            # update val_lens correspondingly
            val_lens[i] = len(q_value) + 2

        # longest question in current batch
        max_len = max(val_lens)

        # then generate zeros array with size(num_of_q * max_len * word_dim)
        val_emb_array = np.zeros((B, max_len, self.N_word), dtype=np.float32)
        # print("val_embs",val_embs)
        # print(val_emb_array)
        for i in range(B):
            for j in range(len(val_embs[i])):
                # : might be removed
                # val_emb_array[i, j, :] = val_embs[i, j]
                # print(val_emb_array[i,j])
                val_emb_array[i, j] = val_embs[i][j]

        # to tensor
        val_input = torch.from_numpy(val_emb_array)
        if self.gpu:
            val_input = val_input.cuda()
        val_input_var = Variable(val_input)

        return val_input_var, val_lens
        

    def gen_col_batch(self, cols):
        '''
        similar as gen_x_batch
        input:
            cols: array of table columns with size the number of batch size
        '''
        col_len = np.zeros(len(cols), dtype=np.int64)

        names = []
        # append column names of x tables 
        for i, one_cols in enumerate(cols):
            names = names + one_cols
            col_len[i] = len(one_cols)
        

        # names is[[tb1_col1],[tb1_col2],...], all column names in this batch 
        # print("names:"+ str(names))
        # col_len is [5,7,...] size batch size; each col number of words for each table
        # print("lens:" + str(col_len))

        name_input_var, name_len = self.str_list_to_batch(names)
        return name_input_var, name_len, col_len

    def str_list_to_batch(self, str_list):
        # B: number of cols of all tables in this batch
        B = len(str_list)
        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, one_str in enumerate(str_list):
            # get word_emb for each cols; if does not exist, fill with 0
            val = [self.word_emb.get(x, np.zeros(self.N_word, dtype=np.float32)) for x in one_str]
            val_embs.append(val)
            val_len[i] = len(val)
        max_len = max(val_len)

        val_embs_array = np.zeros((B, max_len, self.N_word), dtype=np.float32)
        for i in range(B):
            for j in range(len(val_embs[i])):
                val_embs_array[i, j, :] = val_embs[i][j]
        val_input = torch.from_numpy(val_embs_array)
        if self.gpu:
            val_input = val_input.cuda()
        val_input_var = Variable(val_input)

        return val_input_var, val_len   


