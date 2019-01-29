import numpy as np
import argparse
from scipy.spatial import distance


def calc(target, vectors, topn=10):
    # Calc distance
    distances = distance.cdist([target], vectors, "cosine")[0]
    min_index = np.argmin(distances)
    min_distance = distances[min_index]
    max_similarity = 1 - min_distance
    print(min_index, min_distance, max_similarity)
    print(tup[min_index][0])

    return
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='dir for sanity check files')
    parser.add_argument('--word', type=str, help='words for sanity check')
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
    # print(full[0][0])
    # print(full[0].dtype)
    # print(full.shape, full.dtype)

    # Get one vec based on word

    target = [v for i, v in enumerate(tup) if v[0] == args.word][0][1]
    # Convert to np
    target = np.array([float(item) for item in target.split(' ')])
    # print(target.shape, target.dtype)
    # print(target)

    calc(target, vectors, topn=10)

