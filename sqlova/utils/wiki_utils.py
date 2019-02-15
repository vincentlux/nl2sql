import os, json
import torch

def load_wiki(path_wiki, toy_model, toy_size, no_w2i=True, no_hs_tok=True):
    train_data, train_table = load_wiki_by_mode(path_wiki, mode="train", toy_model=toy_model, toy_size=toy_size, no_hs_tok=no_hs_tok)
    dev_data, dev_table = load_wiki_by_mode(path_wiki, mode="dev", toy_model=toy_model, toy_size=toy_size, no_hs_tok=no_hs_tok)

    return train_data, train_table, dev_data, dev_table

def load_wiki_by_mode(path_wiki, mode="train", toy_model=False, toy_size=10, no_hs_tok=False):
    # Load training and dev set by setted mode
    path_sql = os.path.join(path_wiki, mode+"_tok.jsonl")
    if no_hs_tok:
        path_tab = os.path.join(path_wiki, mode+".tables.jsonl")
    else:
        path_tab = os.path.join(path_wiki, mode+"_tok.tables.jsonl")
    
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
    
    with open(path_tab) as f:
        for idx, line in enumerate(f):
            if toy_model and idx > toy_size:
                break
            tab = json.loads(line.strip())
            table[tab["id"]] = tab
            """
            t1_table:  {'header': ['Player', 'No.', 'Nationality', 'Position', 'Years in Toronto', 'School/Club Team'], 
            'page_title': 'Toronto Raptors all-time roster', 'types': ['text', 'real', 'text', 'text', 'text', 'text'], 
            'id': '1-10015132-7', 'section_title': 'G', 'caption': 'G', 
            'rows': [['Sundiata Gaines', 2, 'United States', 'Guard', '2011', 'Georgia'], 
            ['Jorge Garbajosa', 15, 'Spain', 'Forward', '2006-08', 'CB Málaga (Spain)'], 
            ['Chris Garner', 0, 'United States', 'Guard', '1997-98', 'Memphis'], 
            ['Rudy Gay', 22, 'United States', 'Forward', '2013-present', 'Connecticut'], 
            ['Dion Glover', 22, 'United States', 'Guard', '2004', 'Georgia Tech'], 
            ['Joey Graham', 14, 'United States', 'Guard-Forward', '2005-09', 'Oklahoma State']], 
            'name': 'table_10015132_7'}
            """
    return sql, table

    def get_loader_wiki(data_train, data_dev, batch_size, shuffle_train=True, shuffle_dev=False):
        train_loader = torch.utils.data.DataLoader(
            batch_size=batch_size,
            dataset=data_train,
            shuffle=shuffle_train,
            num_workers=4, # Why?
            collate_fn=lambda x:x   # Why?
        )

        dev_loader = torch.utils.data.DataLoader(
            batch_size=batch_size,
            dataset=data_dev,
            shuffle=shuffle_dev,
            num_workers=4, # Why?
            collate_fn=lambda x:x   # Why?
        )
        return train_loader, dev_loader