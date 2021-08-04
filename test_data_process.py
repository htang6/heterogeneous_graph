from data_process import *

f = open('../data/cl.obo.txt', 'r')
content = f.read()
term_blocks = content.split('\n\n')
test_block = term_blocks[3]

def test_parse_id():
    result = parse_ID(test_block)
    assert result == 'BFO:0000004'

def test_parse_name():
    result = parse_name(test_block)
    assert result == 'independent continuant'

def test_parse_syno():
    result = parse_syno(test_block)
    assert len(result[0]) == 0
    assert len(result[1]) == 0

def test_parse_rela():
    result = parse_relation(test_block)
    assert len(result) == 1
    assert result[0][0] == 'BFO:0000050'
    assert result[0][1] == 'BFO:0000004'

def test_parse_main_rela():
    result = parse_main_relation(test_block)
    assert len(result) == 3
    assert result[0][0] == 'is_a'
    assert result[0][1] == 'BFO:0000002'
    assert result[1][0] == 'disjoint_from'
    assert result[1][1] == 'BFO:0000020'
    assert result[2][0] == 'disjoint_from'
    assert result[2][1] == 'BFO:0000031'

