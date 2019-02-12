# L1 and L2 normalization foMr embeddings

from sklearn import preprocessing
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--i", type=str, default="", help="name of input embeddings under /embeddings")
parser.add_argument("--norm", type=str, default="l2", help="l1 or l2")
parser.add_argument("--o", type=str, default="", help="name of output embeddings under /embeddings")
args = parser.parse_args()

f = pd.read_csv('embeddings/'+args.i, sep=" ", header = None)
word = f[[0]]
f.drop(0, axis=1, inplace=True)
np_f = f.values
np_f_norm = preprocessing.normalize(np_f, norm=args.norm)
np_df = pd.DataFrame(data=np_f_norm)

word = word.reset_index(drop=True)
np_df = np_df.reset_index(drop=True)
result = pd.concat([word, np_df],axis=1)
result.to_csv('embeddings/'+args.o, sep=" ", index=False, header=False, encoding="utf-8")

