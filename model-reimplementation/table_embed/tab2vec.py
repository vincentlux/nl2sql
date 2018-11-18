'''
code snippets refer to sqlnet
train_tok.tables.jsonl
header_tok
rows (need to be tokenized)
'''
import json

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
        v = {target: v[target] for target in ["header_tok", "rows"]}
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
    # val_sql_data, val_table_data = load_data('data/dev_tok.jsonl',
    #         'data/dev_tok.tables.jsonl', use_small=use_small)

    return sql_data, table_data


def intersect(table_data):
    '''
    input: 
    table_dictionary: nested dict, {id:{'header_tok':array1,'columns'}}
    output: 2d array saving each header-col pair at same dim
    '''
    full_list = []
    for k, v in table_data.items():
        for k2, v2 in v.items():
            


    return

sql_data, table_data = load_dataset(use_small=True)
print(len(table_data))
print(table_data['1-1000181-1'])
