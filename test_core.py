import torch
from bern_corrupter import BernCorrupter
from utils import mrr_mr_hitk

def test_bern_corrupter():
    test_data = [
        [1, 1, 1],
        [1, 2, 3],
        [0, 0, 0]
        ]

    corrupter = BernCorrupter(test_data, 1, 1)
    assert corrupter.bern_prob[0] == 0.75

def test_metrics():
    scores = []
    idx = range(0, 20)
    total = sum(idx)
    frac = 1/total
    for i in idx:
        scores.append(frac*i)
    scores = torch.FloatTensor(scores)

    target1 = 0
    rr, r, hits = mrr_mr_hitk(scores, target1, descend=True)
    assert rr == 0.05 and r == 20
    assert hits[0] == 0 and hits[1] == 0 and hits[2] == 0

    target2 = 10
    rr, r, hits = mrr_mr_hitk(scores, target2, descend=True)
    assert rr == 0.1 and r == 10
    assert hits[0] == 0 and hits[1] == 0 and hits[2] == 1

    target3 = 19
    rr, r, hits = mrr_mr_hitk(scores, target3, descend=True)
    assert rr == 1 and r == 1
    assert hits[0] == 1 and hits[1] == 1 and hits[2] == 1




