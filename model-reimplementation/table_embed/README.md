# Run

`python table2vec.py`


`python table2vec.py --suffix <str>`: save trained embeddings with input suffix


# Idea
1. To enhance relationship between column info(possibly n-gram) and headers (similar to [ELMo](https://arxiv.org/pdf/1802.05365.pdf))
2. Consider length of column info as a factor 
3. __Learn more about transformer, self-attention__


# Method
* To generate table embeddings and test performance \
    "header1","info1-1","header1","info1-2"
    "header2","info2-1","header2","info2-2"  

# Pytorch
[gpu](https://morvanzhou.github.io/tutorials/machine-learning/torch/5-02-GPU/)

[batch normalization](https://morvanzhou.github.io/tutorials/machine-learning/torch/5-04-batch-normalization/)

[Deep Q Network](https://morvanzhou.github.io/tutorials/machine-learning/torch/4-05-A-DQN/)