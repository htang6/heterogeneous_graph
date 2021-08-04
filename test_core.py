from bern_corrupter import BernCorrupter

def test_bern_corrupter():
    test_data = [
        [1, 1, 1],
        [1, 2, 3],
        [0, 0, 0]
        ]

    corrupter = BernCorrupter(test_data, 1, 1)
    assert corrupter.bern_prob[0] == 0.75