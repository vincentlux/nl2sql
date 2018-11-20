import os
import utils
import logging
from gensim.models import Word2Vec

def tab_to_vec():

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)
    # Set values for various parameters
    num_features = 300    # Word vector dimensionality                      
    min_word_count = 5   # Minimum word count                        
    num_workers = 4       # Number of threads to run in parallel
    context = 5           # Context window size                                                                                    
    downsampling = 0   # Downsample setting for frequent words

    _sql_data, table_data = utils.load_dataset()
    sentences = utils.intersect(table_data)

    print('training model...')
    model = Word2Vec(sentences, workers=num_workers, \
                size=num_features, min_count = min_word_count, \
                window = context, sample = downsampling)
    if not os.path.exists("./embeddings"):
        os.makedirs("./embeddings")
    model_name = "./embeddings/300features_5minwords_5context.txt"
    model.wv.save_word2vec_format(model_name, binary=False)
    # next to do: save model as vectors.txt


tab_to_vec()
