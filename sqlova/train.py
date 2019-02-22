# sqlova reimplementation with modification
# largely refers to sqlova from Wonseok Hwang

import os, sys, argparse, re

import torch 
import torch.nn as nn
import torch.nn.functional as F

from utils.wiki_utils import load_wiki, get_loader_wiki

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data(path_wiki, args):
    train_data, train_table, dev_data, dev_table = load_wiki(path_wiki,\
                                                         args.toy_model, args.toy_size, no_w2i=True, no_hs_tok=True)
    # get torch data_loader
    train_loader, dev_loader = get_loader_wiki(train_data, dev_data, args.bs, shuffle_train=True)
    return train_data, train_table, dev_data, dev_table, train_loader, dev_loader

def get_models(args, bert_pt_path):
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_ops = ['=', '>', '<', 'OP']

    print(f"Batch_size = {args.bs * args.accumulate_gradients}")
    print(f"BERT parameters:")
    print(f"learning rate: {args.lr_bert}")
    print(f"Fine-tune BERT: {args.fine_tune}")



if __name__ == "__main__":
    if torch.cuda.is_available():
        print("training using gpu")
    # 1. hyper-param
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=200, type=int)
    parser.add_argument("--bs", default=32, type=int, help="Batch size")
    parser.add_argument('--no_pretraining', action='store_true', help='Use BERT pretrained model') # default value: false
    parser.add_argument("--accumulate_gradients", default=1, type=int,
                        help="The number of accumulation of backpropagation to effectivly increase the batch size.")
    
    
    args = parser.parse_args()

    ## 2. path
    path = "./"
    path_wiki = os.path.join(path, "data", "wikisql_tok")
    bert_pt_path = path_wiki

    path_eval = './'

    ## load data
    train_data, train_table, dev_data, dev_table, train_loader, dev_loader = get_data(path_wiki, args)
    print("Finished loading data")

    # 4. Build and load models


