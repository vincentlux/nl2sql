import numpy as np
import argparse
from scipy.spatial import distance


def calc(target, vectors, topn=10):
    # Calc distance
    distances = distance.cdist([target], vectors, "cosine")[0]
    min_indices = np.argpartition(distances, topn)[:topn]
    min_indices = min_indices[np.argsort(distances[min_indices])]
    for i in range(topn):
        min_distance = distances[min_indices[i]]
        max_similarity = 1 - min_distance
        print(tup[min_indices[i]][0], max_similarity)

    min_index = np.argmin(distances)
    min_distance = distances[min_index]
    max_similarity = 1 - min_distance
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='dir for sanity check files')
    parser.add_argument('--word', type=str, help='words for sanity check')
    parser.add_argument('--topn', type=int, help='topn most similar words')
    args = parser.parse_args()

    # Read in as tuple and later fetch by corrsponding index
    tup = []
    with open(args.input) as f:
        print('Start processing...')
        for line in f:
            temp = (line.split(' ', 1)[0], line.split(' ', 1)[1].rstrip())
            tup.append(temp)

    # Convert tup[x][1] into nparray
    vectors = []
    for i in range(len(tup)):
        one = [float(item) for item in [tup[i][1]][0].split(' ')]
        vectors.append(one)
    vectors = np.array(vectors, dtype=np.float)

    # Get one vec based on word
    try:
        target = [v for i, v in enumerate(tup) if v[0] == args.word][0][1]
        # Convert to np
        target = np.array([float(item) for item in target.split(' ')])

        calc(target, vectors, topn=args.topn)
    except Exception as e:
        print(e)
