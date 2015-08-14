# Entropy estimates of clusters.
from __future__ import division
import numpy as np
cimport numpy as np
cimport cython
import itertools
import entropy

def triplets_probabilities( np.ndarray[dtype=np.float_t,ndim=2] v ):
    cdef np.ndarray[dtype=np.float_t,ndim=2] tripletsP
    cdef np.ndarray[dtype=bool,ndim=1] ix
    cdef tuple triplet
    cdef np.ndarray[dtype=np.float_t,ndim=1] state
    tripletsP = np.zeros((v.shape[1]*(v.shape[1]-1)*(v.shape[1]-2)/6,v.shape[1]))

    tripletsGenerator = itertools.combinations(range(v.shape[1]),3)
    binStates = entropy.bin_states(3,sym=True)
    for triplet in tripletsGenerator:
        ix = np.prod( v[:,triplet]!=0,1 )==1
        tripletsP.append(np.array([ np.sum(np.sum(v[:,triplet]==state[None,:],1)==3) 
                                 for state in binStates ])/np.sum(ix))
    return np.array(tripletsP)
