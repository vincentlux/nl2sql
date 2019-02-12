import os
import argparse
import utils as u
import table2vec as tv
def hyper_param(parser):
    parser.add_argument("--suffix", type=str, default="", help="Suffix at the end of the model name.")
    parser.add_argument("--feature", type=int, default=300, help="number of features")
    parser.add_argument("--wordcount", type=int, default=5, help="minimum occurence")
    parser.add_argument("--window", type=int, default=5, help="size of sliding window")
    parser.add_argument("--downsample", type=float, default=0.0, help="downsampling")
    
    parser.add_argument("--ngram", type=bool, default=True, help="generate header as n_gram or not")
    
    args = parser.parse_args()
    return args





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = hyper_param(parser)
    
    table_data = u.load_dataset()
    sentences = u.intersect(table_data, args)
    model = tv.tab_to_vec(sentences, args.feature, args.wordcount, args.window, args.downsample)
    if not os.path.exists("./embeddings"):
        os.makedirs("./embeddings")
    model_name = "./embeddings/"+str(args.feature)+"f_"+str(args.wordcount)+ \
                "minw_"+str(args.window)+"cont_tdt_"+args.suffix+".txt"
    model.wv.save_word2vec_format(model_name, binary=False)
