from entropy import *

def test_convert_to_maj():
    """Some test for {-1,1} case."""
    X=np.array([[1,1,1,1],[-1,-1,1,1],[-1,-1,-1,1]])
    majX=convert_to_maj(X)
    assert np.array_equal(majX.sum(1),np.array([4,0,2]))
    assert np.array_equal(majX[1],X[1])

    majX=convert_to_maj(X,maj0or1=0)
    assert np.array_equal(majX.sum(1),-np.array([4,0,2]))
    assert np.array_equal(majX[1],X[1])


