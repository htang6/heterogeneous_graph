from data_process import *

f = open('./data/test.txt', 'r')
content = f.read()
term_blocks = content.split('\n\n')
test_block = term_blocks[0]

def test_parse_id():
    result = parse_ID(test_block)
    assert result == 'UBERON:0001311'

def test_parse_name():
    result = parse_name(test_block)
    assert result == 'inferior vesical artery'

def test_parse_syno():
    result = parse_syno(test_block)
    assert len(result[0]) == 0
    assert len(result[1]) == 0

def test_parse_rela():
    result = parse_relation(test_block)
    assert len(result) == 6
    assert result[0][0] == 'BFO:0000050'
    assert result[0][1] == 'UBERON:0001309'
    assert result[1][0] == 'RO:0002178'
    assert result[1][1] == 'UBERON:0000998'

def test_parse_main_rela():
    result = parse_main_relation(test_block)
    assert len(result) == 2
    assert result[0][0] == 'is_a'
    assert result[0][1] == 'UBERON:0004573'
    assert result[1][0] == 'is_a'
    assert result[1][1] == 'UBERON:0009027'

if __name__ == '__main__':
    test_parse_rela()
