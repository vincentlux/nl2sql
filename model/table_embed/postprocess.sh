#!/usr/bin/env bash
embdir='./embeddings/'
embname=${1}

# echo $embdir$embname

# remove first line
# echo "$(tail -n +2 $embdir$embname)" > $embdir$embname

# send to longleaf
scp $embdir$embname xiaopeng@longleaf.unc.edu:/nas/longleaf/home/xiaopeng/project/nl2sql/nl2sql/model/re_SQLNet_t2v/glove/embeddings
