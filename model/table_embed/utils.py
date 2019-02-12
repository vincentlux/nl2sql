'''
code snippets refer to sqlnet
train_tok.tables.jsonl
header_tok
rows (need to be tokenized)
'''
import json
import re

def load_data(sql_paths, table_paths, use_small=False):
    if not isinstance(sql_paths, list):
        sql_paths = (sql_paths, )
    if not isinstance(table_paths, list):
        table_paths = (table_paths, )
    sql_data = []
    table_data = {}

    max_col_num = 0
    for SQL_PATH in sql_paths:
        print("Loading data from %s"%SQL_PATH)
        with open(SQL_PATH) as inf:
            for idx, line in enumerate(inf):
                # use_small only take 1000 lines
                if use_small and idx >= 100:
                    break
                sql = json.loads(line.strip())
                sql_data.append(sql)

    for TABLE_PATH in table_paths:
        print("Loading data from %s"%TABLE_PATH)
        with open(TABLE_PATH) as inf:
            for line in inf:
                tab = json.loads(line.strip())
                table_data[tab['id']] = tab
    # only get header_tok and rows (transposed)
    for k, v in table_data.items():
        v = {target: v[target] for target in ["header", "rows"]}
        table_data[k] = v
        for k2, v2 in v.items():
            # transpose
            if k2 == "rows":
                v[k2] = [list(x) for x in zip(*v2)]
                table_data[k] = v
        v["columns"] = v.pop("rows")

    for sql in sql_data:
        assert sql['table_id'] in table_data
    return sql_data, table_data

def load_dataset(use_small=False):
    print("Loading from original dataset")
    sql_data, table_data = load_data('data/train_tok.jsonl',
            'data/train_tok.tables.jsonl', use_small=use_small)
    sql_data_dev, table_data_dev = load_data('data/dev_tok.jsonl',
            'data/dev_tok.tables.jsonl', use_small=use_small)
    sql_data_test, table_data_test = load_data('data/test_tok.jsonl',
            'data/test_tok.tables.jsonl', use_small=use_small)

    # Use train,dev,test tables for word2vec training
    table_data_all = {**table_data, **table_data_dev, **table_data_test}
    print(len(table_data_all))
    return table_data_all


def intersect(table_data, args):
    '''
    input:
    table_data: nested dict, {id:{'header_tok':array1,'columns'}}
    output: 2d array saving each header-col pair at same dim
    '''
    header_list = []
    column_list = []
    intersect_list = []
    # Save into two lists
    for k, v in table_data.items():
        for k2, v2 in v.items():
            if k2 == "header":
                # if n_gram header => convert to  "header name" to "header_name"
                if args.ngram:
                    v2 = [re.sub(r'\s+', '_',  i.strip()) for i in v2]
                    header_list.append(v2)
                    # print(header_list)
                else:
                    # print("header?",v2)
                    header_list.append(v2)
            else:
                column_list.append(v2)
    # Loop over and merge(intersect) headers and columns
    for tab in range(len(column_list)):
        for col in range(len(column_list[tab])):
            dup_list = [header_list[tab][col]] * len(column_list[tab][col])
            intersect = [(dup_list[i], column_list[tab][col][i]) for i in range(0, len(dup_list))]
            # flatten tuple
            intersect = [item for sublist in intersect for item in sublist]
            intersect_list.append(intersect)
    # lowercase, remove punc, tokenize
    for line in range(len(intersect_list)):
        for element in range(len(intersect_list[line])):
            intersect_list[line][element] = re.sub(r"[^\w]"," ", str(intersect_list[line][element]).lstrip('_').rstrip('_')).lower().split()

        intersect_list[line] = [item for sublist in intersect_list[line] for item in sublist]
        if line % 10000 == 0:
            print("processed "+str(line)+" of "+ str(len(intersect_list))+" lines")

    # print(intersect_list)
    return intersect_list


# table_data = load_dataset(use_small=True)

# intersect_list = intersect(table_data)
# print(len(intersect_list))
# print(intersect_list[:10])
# next to do: tokenize each object and feed into word2vec
