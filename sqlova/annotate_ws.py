from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os
import records
import ujson as json
from stanza.nlp.corenlp import CoreNLPClient
from tqdm import tqdm
import copy
from wikisql.lib.common import count_lines

client = None

def annotate(sentence, lower=True):
    global client
    if client is None:
        client = CoreNLPClient(default_annotators='ssplit,tokenize'.split(','))
    words, gloss, after = [], [], []
    for s in client.annotate(sentence):
        for t in s:
            words.append(t.word)
            gloss.append(t.originalText)
            after.append(t.after)
    if lower:
        words = [w.lower for w in words]
    return {
        'gloss': gloss,
        'words': words,
        'after': after,
    }

def find_sub_list(sl, l):
    # from stack overflow.
    results = []
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind + sll] == sl:
            results.append((ind, ind + sll - 1))

    return results

def check_wv_tok_in_nlu_tok(wv_tok1, nlu_t1):
    g_wvi1_corenlp = []
    nlu_t1_low = [tok.lower() for tok in nlu_t1]
    for i_wn, wv_tok11 in enumerate(wv_tok1):
        wv_tok11_low = [tok.lower() for tok in wv_tok11]
        # print("wv_tok11_low: ",wv_tok11_low)
        # print("nlu_t1_low: ",nlu_t1_low)
        results = find_sub_list(wv_tok11_low, nlu_t1_low)
        st_idx, ed_idx = results[0]

        g_wvi1_corenlp.append( [st_idx, ed_idx] )

    return g_wvi1_corenlp


def annotate_example_ws(example, table):
    ann = {'table_id': example['table_id'], 'phase':example['phase']}
    _nlu_ann = annotate(example['question'])
    ann['question'] = example['question']
    ann['question_tok'] = _nlu_ann['gloss']

    ann['sql'] = example['sql']
    ann['query'] = sql = copy.deepcopy(example['sql'])

    conds1 = ann['sql']['conds']
    wv_ann1 = []
    for conds11 in conds1:
        _wv_ann1 = annotate(str(conds11[2]))
        wv_ann11 = _wv_ann1['gloss']
        # print(wv_ann11)
        wv_ann1.append(wv_ann11)

    try:
        wvi1_corenlp = check_wv_tok_in_nlu_tok(wv_ann1, ann['question_tok'])
        ann['wvi_corenlp'] = wvi1_corenlp
    except:
        ann['wvi_corenlp'] = None
        ann['tok_error'] = 'SQuAD style st, ed not found under CoreNLP'

    return ann    



if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--din', default='./data/WikiSQL-1.1', help='data directory')
    parser.add_argument('--dout', default='./data/wikisql_tok', help='output directory')
    parser.add_argument('--aug_type', type=str, default='mt', help='mt or pos or reverse')
    parser.add_argument('--out_name', type=str, default='default', help='notes for different file name')
    parser.add_argument('--remove_err_tok', action='store_true', help="If present, remove tokens which do not have a match with sql table")
    args = parser.parse_args()


    for split in ['train','dev','test']:
        if split == 'train':
            fsplit = os.path.join(args.din, split)+'_'+args.aug_type+'.jsonl'
            ftable = os.path.join(args.din, split)+'.tables.jsonl'
            fout = os.path.join(args.dout, split)+'_'+args.aug_type+'_'+args.out_name+ '_tok.jsonl'
        else:
            fsplit = os.path.join(args.din, split) + '.jsonl'
            ftable = os.path.join(args.din, split) + '.tables.jsonl'
            fout = os.path.join(args.dout, split) + '_tok.jsonl'
        
        print(f'Annotating {fsplit}')
        with open(fsplit) as fs, open(ftable) as ft, open(fout, 'wt') as fo:
            print('loading tables')
            # print(ftable)
            tables = {}
            for line in tqdm(ft, total=count_lines(ftable)):
                # print(line)
                d = json.loads(line)
                tables[d['id']] = d
            print("loading examples")
            n_written = 0
            count = -1
            for line in tqdm(fs, total=count_lines(fsplit)):
                count += 1
                d = json.loads(line)
                a = annotate_example_ws(d, tables[d['table_id']])

                if split == 'train' and args.remove_err_tok:
                    try:
                        a["tok_error"]
                        pass
                    except:
                        fo.write(json.dumps(a) + '\n')
                        n_written += 1
                else:
                    fo.write(json.dumps(a) + '\n')
                    n_written += 1

            print(f'wrote {n_written} examples')
