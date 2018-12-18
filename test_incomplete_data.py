from .incomplete_data import *
from coniii.utils import pair_corr
np.random.seed(0)


def test_check_triplet():
    # simple agreement case
    for i in range(10):
        X = np.random.choice([0,1], size=(100,3))
        si, sisj = pair_corr(X)
        assert check_triplet(X) and check_triplet((si,sisj))

    # simple case where two pairwise subsets agree all the time but third doesn't
    X = np.array([[1,1,0],
                  [1,0,1],
                  [0,-1,1]])
    si = np.array([1,.5,1])
    sisj = np.array([1,1,0])
    
    assert check_triplet(X)==False and check_triplet((si,sisj))==False
