# First test with google api
import re, os, copy
import argparse
import ujson as json
from time import sleep
from google.cloud import translate


def load_wiki(path_wiki, fout, mode="train"):
    # Load training and dev set by setted mode
    path_sql = os.path.join(path_wiki, mode+".jsonl")
    sql = []
    table = {}
    attempts = 0
    with open(path_sql) as f:
        for idx, line in enumerate(f):
            s = json.loads(line.strip().replace("\xc2\xa0", " ")) # deal with utf8 bytes
            sql.append(s)
            trans = copy.deepcopy(s)

            while attempts < 10:
                try:
                    trans["question"] = back_trans(trans["question"])
                    sql.append(trans)
                    break
                except:
                    sleep(5)
                    attempts += 1
            print(idx)
            if idx % 10 == 0:
                with open(fout, 'at') as f:
                    for line in sql:
                        # print("line: ", str(line))
                        # a = json.loads(str(line))
                        f.write(json.dumps(line) + "\n")
                        sql = []
            """
            {'phase': 1, 'table_id': '1-1570003-2', 'question': 'What was the playoff advancement during the year 1998?', 'sql': {'sel': 4, 'conds': [[0, 0, 1998]], 'agg': 0}}
            """
    return sql

def back_trans(str_in):
    tc = translate.Client()
    fr = tc.translate(str_in, target_language="fr")
    en = tc.translate(fr["translatedText"], target_language="en")
    return en["translatedText"]


if __name__ == "__main__":
    # preprocess
    path_in = "../data/WikiSQL-1.1"
    path_out = "../data/mt"


    fout = os.path.join(path_out, 'train_mt') + '.jsonl'
    sql = load_wiki(path_in, fout, mode="train")

    # with open(fout, 'wt') as f:
    #     for line in sql:
    #         # print("line: ", str(line))
    #         # a = json.loads(str(line))
    #         f.write(json.dumps(line) + "\n")
        




    # print()
