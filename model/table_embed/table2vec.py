import os
import utils
import logging
import argparse
from gensim.models import Word2Vec

def tab_to_vec(sentences, num_features, min_word_count, context, downsampling):
    '''
    Train table embeddings and save under ./embeddings folder
    '''
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)
    # Set values for various parameters
    # num_features = 300    # Word vector dimensionality                      
    # min_word_count = 5   # Minimum word count 
    # context = 5           # Context window size                                                                                    
    # downsampling = 0   # Downsample setting for frequent words                       
    num_workers = 4       # Number of threads to run in parallel


    print('training model...')
    model = Word2Vec(sentences, workers=num_workers, \
                size=num_features, min_count = min_word_count, \
                window = context, sample = downsampling)
    return model


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--suffix", type=str, default="", help="Suffix at the end of the model name.")
#     parser.add_argument("--feature", type=int, default=300, help="number of features")
#     parser.add_argument("--wordcount", type=int, default=5, help="minimum occurence")
#     parser.add_argument("--window", type=int, default=5, help="size of sliding window")
#     parser.add_argument("--downsample", type=float, default=0.0, help="downsampling")
#     args = parser.parse_args()

#     table_data = utils.load_dataset()
#     sentences = utils.intersect(table_data)
#     model = tab_to_vec(sentences, args.feature, args.wordcount, args.window, args.downsample)
#     if not os.path.exists("./embeddings"):
#         os.makedirs("./embeddings")
#     model_name = "./embeddings/"+args.feature+"features_"+args.wordcount+ \
#                 "minwords_"+args.window+"context" + args.suffix + ".txt"
#     model.wv.save_word2vec_format(model_name, binary=False)
    
