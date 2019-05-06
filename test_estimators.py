# ===================================================================================== #
# Testing module for entropy estimators.
# Author: Eddie Lee, edlee@alumni.princeton.edu
# ===================================================================================== #
from .estimators import *


def test_cross_entropy():
    X = np.array([[1,1,1],[1,1,-1]])
    Y = X.copy()
    assert cross_entropy(X,Y)==0

    Y[0,0]*=-1
    assert np.isnan(cross_entropy(X,Y))

    Y[0,0]*=-1
    X[0,0]*=-1
    assert cross_entropy(X,Y)

def test_S_naive():
    for n in range(1,6):
        assert S_naive(np.arange(2**n))==n

def test_S_poly():
    np.random.seed(0)
    X = np.random.randint(0, 2, size=100_000)

    assert abs(S_poly(X, [.8,.95,1], 10)-1)<1e-3
