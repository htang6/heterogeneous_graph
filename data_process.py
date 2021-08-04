from collections import defaultdict
import re

'''
I need to generate following info:
edge list (n1, n2, edge_info)
node list (cid, node_info)
'''
def parse_block(block):
    cid = parse_ID(block)
    if cid == None:
        return None

    name = parse_name(block)
    syno = parse_syno(block)
    main_relations = parse_main_relation(block)
    other_relations = parse_relation(block)

    return {
        'cid': cid,
        'name': name,
        'synonyms': syno,
        'main_relations': main_relations,
        'other_relations': other_relations
    }


def parse_ID(block):
    id_regx = 'id: (.*)'
    match_id = re.search(id_regx, block)
    cID = None
    if match_id is not None:
        cID = match_id.group(1)
    return cID

def parse_name(block):
    name_regx='name: (.*)'
    match_name = re.search(name_regx, block)
    pName = None
    if match_name is not None:
        pName = match_name.group(1)
    return pName

def parse_syno(block):
    synonym_regx = '\"(.*)\" (EXACT|RELATED)'
    match_synonym_list = re.findall(synonym_regx, block)
    exact_list = []
    related_list = []
    for synonym in match_synonym_list:
        if synonym[1] == 'EXACT':
            exact_list.append(synonym[0])
        elif synonym[1] == 'RELATED':
            related_list.append(synonym[0])
    return exact_list, related_list

def parse_relation(block):
    rela_regx = 'relationship: (.*) (.*) \!'
    match_rela_list = re.findall(rela_regx, block)
    result = []
    for rela in match_rela_list:
        result.append((rela[0], rela[1]))
    return result

def parse_main_relation(block):
    rela_regx0 = 'is_a: ([A-Z]+:[0-9]+) .*\!'
    rela_regx1 = 'disjoint_from: ([A-Z]+:[0-9]+) .*\!'

    result = []
    match_rela_list = re.findall(rela_regx0, block)
    for rela in match_rela_list:
        result.append(('is_a', rela))
    match_rela_list = re.findall(rela_regx1, block)
    for rela in match_rela_list:
        result.append(('disjoint_from', rela))
    
    return result
    

def preprocess(phrase):
    pass
    
