# First test with google api
import re, os
import argparse
from googletrans import Translator


def load_wiki(path_wiki, mode="train"):
    # Load training and dev set by setted mode
    path_sql = os.path.join(path_wiki, mode+".jsonl")

    sql = []
    table = {}
    with open(path_sql) as f:
        for idx, line in enumerate(f):
            if toy_model and idx >= toy_size:
                break
            s = json.loads(line.strip())
            sql.append(s)
            """
            {'table_id': '1-1000181-1', 'phase': 1, 'question': 'Tell me what the notes are for South Australia ', 
            'question_tok': ['Tell', 'me', 'what', 'the', 'notes', 'are', 'for', 'South', 'Australia'], 
            'sql': {'sel': 5, 'conds': [[3, 0, 'SOUTH AUSTRALIA']], 'agg': 0}, 
            'query': {'sel': 5, 'conds': [[3, 0, 'SOUTH AUSTRALIA']], 'agg': 0}, 'wvi_corenlp': [[7, 8]]}
            """




if __name__ == "__main__":
    # preprocess
    path = "../data/WikiSQL-1.1"
    load_wiki(path_wiki, mode="train")


    translator = Translator()


    # print(translator.translate('veritas lux mea', src='la').text)
