import os
import utils
import logging
import argparse
from gensim.models import Word2Vec

def tab_to_vec(sentences):
    '''
    Train table embeddings and save under ./embeddings folder
    '''
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)
    # Set values for various parameters
    num_features = 300    # Word vector dimensionality                      
    min_word_count = 5   # Minimum word count                        
    num_workers = 4       # Number of threads to run in parallel
    context = 5           # Context window size                                                                                    
    downsampling = 0   # Downsample setting for frequent words

    print('training model...')
    model = Word2Vec(sentences, workers=num_workers, \
                size=num_features, min_count = min_word_count, \
                window = context, sample = downsampling)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", type=str, default="", help="Suffix at the end of the model name.")
    args = parser.parse_args()

    table_data = utils.load_dataset()
    sentences = utils.intersect(table_data)
    model = tab_to_vec(sentences)
    if not os.path.exists("./embeddings"):
        os.makedirs("./embeddings")
    model_name = "./embeddings/300features_5minwords_5context" + args.suffix + ".txt"
    model.wv.save_word2vec_format(model_name, binary=False)
    
