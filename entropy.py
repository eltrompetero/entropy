# Functions necessary for calculating entropies.
# 2013-06-25
from __future__ import division
import numpy as np
#import clusters as cluster
from numba import jit
from misc.utils import unique_rows
from numba import float64


class DiscreteEntropy(object):
    def __init__(self,bins,ndim=None):
        """
        Class for entropy estimation.
        2016-12-08
        
        Params:
        -------
        """
        self.bins = bins
        if ndim is None:
            try:
                self.ndim = len(bins)
            except TypeError:
                raise Exception("Must set ndim.")
        else:
            self.ndim = ndim
            self.bins = [bins]*self.ndim
        
    def _estimate_pdf(self,X,bins=None):
        """
        Estimate multi-dimensional entropy.
        """
        if bins is None:
            bins=self.bins
            
        self.pdf,self.edges = np.histogramdd(X,bins=bins)
        self.pdf /= self.pdf.sum()

    def _kde(self,X):
        """
        """
        from sklearn.neighbors import KernelDensity
        kde = KernelDensity(bandwidth=.1,kernel='gaussian')
        kde.fit(X)
        self.kde = kde

    def disc_state(self,X):
        """
        Return a mapping of the discretized states to integers
        2016-12-09

        Params:
        -------
        X (ndarray)
            n_samples x n_dim
        """
        from bisect import bisect
        
        self._estimate_pdf(X)
        disclabel = np.zeros_like(X,dtype=int)
        for i,x in enumerate(X):
            for j in xrange(self.ndim):
                disclabel[i] = [bisect(self.edges[j],x_,hi=len(self.edges[j])-2) for x_ in x]
        return unique_rows(disclabel,return_inverse=True)
        
    def estimate_S(self,X,method='naive',binRange=[]):
        """
        Params:
        -------
        X (array)
            n_samples x n_dim
        method (str='naive')
            'naive': simple estimation using empirical pdf
            'extrapolate': calculate entropy for multiple bin sizes
        binRange (list=[])
            Bounds on the bin range. Bins will be increased on the dimensions from left to right
            consecutively. This must be given for 'extrapolate' method.
        """
        if method=='naive':
            self._estimate_pdf(X)
            self.S = -np.nansum( self.pdf*np.log2(self.pdf) )
        elif method=='extrapolate':
            S = []
            bins=[binRange[0]]*self.ndim
            k = 0
            for i in xrange(binRange[0],binRange[1]):
                if i==binRange[0]:
                    self._estimate_pdf(X,bins)
                    S.append([-np.nansum( self.pdf*np.log2(self.pdf) )])
                    k += 1
                S.append([])
                for j in xrange(self.ndim):
                    bins[j] += 1
                    self._estimate_pdf(X,bins)
                    S[-1].append( -np.nansum( self.pdf*np.log2(self.pdf) ) )
                    k += 1
            self.S = S
        return self.S
# end DiscreteEntropy


def enumerate_states(x):
    """
    Given data array x, assign a unique integer label to each row.
    2016-03-25
    """
    return unique_rows(x,return_inverse=True)

@jit(cache=True)
def bootstrap_MI(data,ix1,ix2,nIters,sampleFraction=1.):
    """
    Use joint_p_mat with bootstrap samples of data to generate error bars on MI.
    2016-01-13

    Params:
    -------
    data (ndarray ndata x ndim)
    ix1 (list)
    ix2 (list)
    nIters (int)
        Number of times to bootstrap
    sampleFraction (float=1)
        fraction of data to sample
    """
    miSamples = np.zeros((nIters))

    for i in xrange(nIters):
        samples = data[np.random.randint(len(data),size=int(sampleFraction*len(data)))]
        p = joint_p_mat(samples,ix1,ix2)
        miSamples[i] = MI(p)
    return miSamples

#@jit(cache=True)
def joint_p_mat(data,ix1,ix2,extra_states=None):
    """
    Return joint probability matrix between different groups of columns of a data matrix. Works on any type of
    data that can be identified separately by misc.utils.unique_rows().
    
    Parameters
    ----------
    data : ndarray
        ndata x ndim
    ix1,ix2 : list
        Index of columns that correspond to groups of columns to compare.
    extra_states : list
        List of states to include into data. This should be a list of two elements. Each element
        will be appended to data[:,ix2] and data[:,ix2].
    """
    assert len(ix1)>=1 and len(ix2)>=1
    
    data1 = data[:,ix1]
    data2 = data[:,ix2]
    if not extra_states is None:
        data1=np.vstack((extra_states[0]))
        data2=np.vstack((extra_states[1]))

    uniq1 = data1[unique_rows(data1)]
    uniq2 = data2[unique_rows(data2)]
    
    p = np.zeros((len(uniq1),len(uniq2)))

    if not extra_states is None:
        # Remove extra states.
        data1=data1[:-len(extra_states[0])]
        data2=data2[:-len(extra_states[1])]

    for row,i in enumerate(uniq1):
        for col,j in enumerate(uniq2):
            p[row,col] = np.logical_and( (data[:,ix1]==i[None,:]).all(1),
                                     (data[:,ix2]==j[None,:]).all(1) ).sum()
    p = p / p.sum()
    return p

def convert_params(h,J,convertTo='01',concat=False):
    """
    Convert parameters from 0,1 formulation to +/-1 and vice versa.
    2014-05-12

    Parameters
    ----------
    h : ndarray
    J : ndarray
    convertTo : str,'01'
        '01' or '11'

    Returns
    -------
    h : ndarray
    J : ndarray
    """
    if len(J.shape)!=2:
        Jmat = squareform(J)
    else:
        Jmat = J
        J = squareform(J)
    
    if convertTo=='11':
        # Convert from 0,1 to -/+1
        Jp = J/4.
        hp = h/2 + np.sum(Jmat,1)/4.
    elif convertTo=='01':
        # Convert from -/+1 to 0,1
        hp = 2.*(h - np.sum(Jmat,1))
        Jp = J*4.

    if concat:
        return np.concatenate((hp,Jp))
    return hp,Jp


def cluster_probabilities(data,order):
    """
    Return probability distributions for subsets of system of binary variables. Given data should be in {-1,1} format where 0's are ignored in computation as non-votes (aren't included in computation of probabilities at all.
    2015-08-14

    Params:
    -------
    data (ndarray)
        k x n matrix with samples x system size
    """
    n = data.shape[1]
    if order==1:
        return np.array([np.array([np.sum(col==-1),np.sum(col==1)])/np.sum(np.logical_or(col==-1,col==1)) 
                         for col in data.T])
    elif order==2:
        pairsP = []
        for i in xrange(n-1):
            for j in xrange(i+1,n):
                nVotesCast = np.sum(np.logical_and(data[:,i]!=0,data[:,j]!=0))
                pairsP.append([ np.sum(np.logical_and(data[:,i]==1,data[:,j]==1)),
                                np.sum(np.logical_and(data[:,i]==1,data[:,j]==-1)),
                                np.sum(np.logical_and(data[:,i]==-1,data[:,j]==1)),
                                np.sum(np.logical_and(data[:,i]==-1,data[:,j]==-1)) ])
                pairsP[-1] = np.array(pairsP[-1]) / nVotesCast
        return np.array(pairsP)
    elif order==3:
        return clusters.triplets_probabilities(data.astype(np.float64))
    return

@jit
def MI(pmat):
    """
    Calculate mutual information between the joint probability distributions in bits. Use with joint_p_mat().
    2015-04-02

    Params:
    -------
    pmat (ndarray)
        2D array
    """
    mi = np.zeros((pmat.size))
    p1 = pmat.sum(1)
    p2 = pmat.sum(0)

    if (p1==0).any():
        return np.nan
    if (p2==0).any():
        return np.nan
    
    k = 0
    for i in xrange(pmat.shape[0]):
        for j in xrange(pmat.shape[1]):
            if pmat[i,j]==0:
                mi[k] = 0 
            else:
                mi[k] = pmat[i,j]*np.log2( pmat[i,j]/(p1[i]*p2[j]) )
            k += 1
    return np.nansum(mi)

def find_state_ix(data,states):
    """
    Inefficient way of getting index of rows of data matrix in given matrix of possible states.
    2015-07-25
    """
    ix = np.zeros((data.shape[0]))
    j = 0
    for i in data:
        ix[j] = np.argwhere( np.sum(i[None,:]==states,1)==states.shape[1] )
        j += 1
    return ix.astype('int')

def dkl(p1,p2,units=2):
    """
    Compute DKL between two probability distributions. Naive.
    dkl = sum( p1*log(p1/p2) )
    
    Params:
    -------
    p1 (ndarray)
    p2 (ndarray)
    units (int=2)
        units for log

    Returns:
    --------
    Dkl
    """
    dkl = p1*np.log(p1/p2)/np.log(units)
    return np.nansum( dkl )

def estimate_S(votes):
    """
    Estimate the entropy of a partially complete data set (missing voters from some
    votes. Fill in the missing votes with the empirical probabilities of those 
    votes occurring in the data.
    2014-05-22

    Params:
    -------
    votes (ndarray)
        array of votes, a vote in each row
    """
    votes,counts = fill_in_vote(votes)
    uVotes = np.array([x for x in set(tuple(x) for x in votes)])
    
    p = np.zeros((uVotes.shape[0]))
    i = 0
    for v in uVotes:
        ix = np.sum(v==votes,1)==votes.shape[1]
        p[i] = np.sum(counts[ix])
        i += 1
    p /= np.sum(p)
    return -np.sum(p*np.log(p)),p

def fill_in_vote(votes):
    """
        For filling in an incomplete data set with missing votes. Take array of votes, 
        and for each vote check whether the vote is complete or not. If it is complete
        then add a count for it. If it isn't complete, go through all possible
        completions of the missing votes given the marginal distributions and give the
        fractional count for each possibility
        Filling in must start with singlets, duplet, tuplet, etc. in order to maintain
        original means.
        NB20140522 
        Args: 
            votes : array of votes, a vote on each row
        Value:
            filledVotes : list of duplets with first element the vote registered and the
                second the fractional count of that vote
    2014-05-26
    """
    from misc_fcns import unique_rows
    
    filledVotes,counts = votes.copy(),np.ones((votes.shape[0]))
    for nanN in range(1,votes.shape[1]):
        for v in filledVotes[np.sum(np.isnan(filledVotes),1)==nanN,:]:
            nanIx = np.argwhere(np.isnan(v)).flatten()
            
            # Just get the complete voting record of the missing people.
            fullVoteIx = np.sum(np.isnan(filledVotes[:,nanIx])==0,1)==nanN
            subVotes = filledVotes[:,nanIx][fullVoteIx,:]
            if subVotes.shape[0]==0:
                raise Exception("There is insufficient data to measure subset.")
            uIx = unique_rows(subVotes)
            uSubVotes = subVotes[uIx,:]
            
            p = get_state_probs( subVotes,uSubVotes, weights=counts[fullVoteIx] )
                        
            # Now, fill the entries of vote.
            toAddVotes,toAddCounts = [],[]
            for i in range(p.size):
                _vote = v.copy()
                _vote[nanIx] = uSubVotes[i,:]
                toAddVotes.append(( _vote ))
                toAddCounts.append(( p[i] ))
            filledVotes = np.vstack(( filledVotes,np.array(toAddVotes) ))
            counts = np.hstack(( counts,np.array(toAddCounts) ))
        # Removed these filled votes.
        keepIx = np.sum(np.isnan(filledVotes),1)!=nanN
        counts = counts[keepIx]
        filledVotes = filledVotes[keepIx,:]

    return filledVotes,counts

def calc_p_J(J,n):
    """
        Get probabilities of states given the J's.
    2014-02-08
    """
    from fast import calc_e

    e = calc_e(J,get_all_states(n))
    expe = np.exp(-e)
    return expe/np.sum(expe)


def get_p_maj(data, allmajstates=None,kmx=None):
    """
        Get probability of k votes in the majority.
        Args:
            data : instance of a staet in each row
            **kargs
            allmajstates : 
            kmx : 
    2014-02-08
    """
    if data.ndim==1:
        if allmajstates==None:
            allmajstates = convert_to_maj( get_all_states(n),maj0or1=0 )
        k = np.sum(allmajstates,1)
        if kmx==None:
            kmx = k.max()+1
        pk = np.zeros((kmx))

        for i in np.arange(kmx):
            pk[i] = np.sum(data[k==i])
    else:
        k,n = np.sum(convert_to_maj(data,maj0or1=0),1), data.shape[1]
        if kmx==None:
            kmx = k.max()+1
        pk = np.zeros((kmx))

        for i in np.arange(kmx):
            pk[i] = np.sum(k==i)
        pk = pk.astype(float)/sum(pk)

    return pk[::-1],np.arange(np.ceil(n/2.),n+1)

def convert_to_maj(states, maj0or1=1):
    """
    Convert given states such that 0 or 1 corresponds to the majority vote. Split
    votes are left as they are.

    Params:
    -------
    states:
    *kwargs
    maj0or1 (bool=True)

    Returns:
    --------
    states
        States converted to majority vote.
    """
    uS = np.unique(states)
    asymCase = np.array_equal( uS,np.array([0.,1.]) ) or (len(uS)==1 and ( uS==0 or uS==1 ))
    symCase = np.array_equal( uS,np.array([-1.,1.]) ) or (len(uS)==1 and ( uS==-1 or uS==1 ))

    if not (asymCase or symCase):
            raise Exception("This function can only deal with {0,1} or {-1,+1}.")
    states = states.copy()
    if symCase:
        states = (states+1)/2

    # In case where we are given one state.
    if states.ndim==1:
        states = np.expand_dims(states,0)
    n = states.shape[1]
    ix = np.sum(states,1) < np.ceil(n/2.)
    states[ix,:] = 1-states[ix,:]

    if maj0or1==0:
        states = 1-states

    if symCase:
        states = states*2-1

    return states

def xbin_states(n,sym=False):
    """Generator for producing binary states.
    Params:
    --------
    n (int)
        number of spins
    sym (bool)
        if true, return {-1,1} basis
    2014-08-26
    """
    assert n>0, "n cannot be <0"
    
    def v():
        for i in xrange(2**n):
            if sym is False:
                yield np.array(list(np.binary_repr(i,width=n))).astype('int')
            else:
                yield np.array(list(np.binary_repr(i,width=n))).astype('int')*2.-1

    return v()

def bin_states(n,sym=False):
    """
    Get all possible binary spin states. Remember that these are uint8 so if
    you want to do arithmetic, must convert to float.
    
    Params:
    -------
    n (int)
        number of spins
    sym (bool)
        if true, return {-1,1} basis
    """
    if n<0:
        raise Exception("n cannot be <0")
    if n>20:
        raise Exception("n is too large to enumerate all states.")
    
    v = np.array([list(np.binary_repr(i,width=n)) for i in range(2**n)]).astype('uint8')

    if sym is False:
        return v
    else:
        return v*2.-1

def get_all_states(n,sym=False):
    """
        Get all possible binary spin states. Remember that these are uint8 so if
        you want to do arithmetic, must convert to float.
        Args:
            n : number of spins
            sym : if true, return {-1,1} basis
    2014-08-26
    """
    import warnings
    warnings.warn("This is deprecated to bin_states(). That name is superior.")
    if n<0:
        raise Exception("n cannot be <0")
    if n>15:
        raise Exception("n is too large to enumerate all states.")
    
    v = np.array([list(np.binary_repr(i,width=n)) for i in range(2**n)]).astype('uint8')

    if sym is False:
        return v
    else:
        return v*2.-1

def squareform(x):
    """
    2014-01-23
    """
    from scipy.spatial import distance
    return distance.squareform(x)

def calc_nan_sisj(data,weighted=None,concat=False):
    """
    Each sample state along a row.

    Params:
    ------
    weighted (np.ndarray,None) : 
        Calculate single and pairwise means given fractional weights for each state in
        the data such that a state only appears with some weight, typically less than
        one
    concat (bool,False)
        return concatenated means if true

    Value:
    ------
    (si,sisj) or sisisj
        duplet of singlet and dubplet means
    2015-06-28
    """
    (S,N) = data.shape
    sisj = np.zeros(N*(N-1)/2)
    
    k=0
    for i in range(N-1):
        for j in range(i+1,N):
            col = data[:,i]*data[:,j]
            sisj[k] = np.nanmean(col)
            k+=1
    
    return (np.nanmean(data,0), sisj)

def xcalc_sisj(data,weighted=None,concat=False):
    """
    Calculate correlations for Ising model using generators.

    Params:
    -------
    data (ndarray)
        (n_samples,n_dim).
    weighted (np.ndarray=None)
        Calculate single and pairwise means given fractional weights for each state in
        the data such that a state only appears with some weight, typically less than
        one.
    concat (bool=False)
        Return concatenated means if true

    Returns:
    --------
    (si,sisj) or np.concatenate((si,sisj))
        duplet of singlet and duplet means
    """
    # Initialize variables.
    if weighted is None:
        def f():
            while True:
                yield 1.
        weighted = f()
    data0 = data.next()
    n = len(data0)
    si,sisj = np.zeros((n)),np.zeros((n*(n-1)//2))
    
    # Initialize loop function that we hope to speed up at some point.
    #@jit(nopython=True)
    def inside_loop(sisj,d,n,thisWeight):
        counter = 0
        for i in xrange(n-1):
            for j in xrange(i+1,n):
                sisj[counter] += d[i]*d[j]*thisWeight
                counter+=1

    def loop(si,sisj,data,weighted):
        ndata = 0
        for d in data:
            thisWeight = weighted.next()
            si += d*thisWeight
            inside_loop(sisj,d,n,thisWeight) 
            ndata += 1
        return ndata
    
    # Calculate correlations.
    thisWeight = weighted.next()
    si += data0*thisWeight
    inside_loop(sisj,data0,n,thisWeight)
    ndata = loop(si,sisj,data,weighted) + 1
    
    try:
        weighted.next()
        si /= ndata
        sisj /= ndata
    except StopIteration:
        pass
            
    # Return values.
    if concat:
        return np.concatenate((si,sisj))
    else:
        return si, sisj

def calc_sisj(data,weighted=None,concat=False, excludeEmpty=False):
    """
    Each sample state along a row.

    Params:
    ------
    weighted (np.ndarray,None) : 
        Calculate single and pairwise means given fractional weights for each state in
        the data such that a state only appears with some weight, typically less than
        one
    concat (bool,False)
        return concatenated means if true
    excludeEmpty (bool,False)
        when using with {-1,1}, can leave entries with 0 and those will not be counted for any pair
        weighted option doesn't do anything here

    Value:
    ------
    (si,sisj) or sisisj
        duplet of singlet and duplet means
    2015-08-10
    """
    S,N = data.shape
    sisj = np.zeros(N*(N-1)//2)
    
    if weighted is None:
        weighted = np.ones((data.shape[0]))/data.shape[0]

    if excludeEmpty:
        assert np.array_equal( np.unique(data),np.array([-1,0,1]) ) or \
            np.array_equal( np.unique(data),np.array([-1,1]) ), "Only handles -1,1 data sets."
        k=0
        for i in xrange(N-1):
            for j in xrange(i+1,N):
                sisj[k] = np.nansum(data[:,i]*data[:,j]) / np.nansum(np.logical_and(data[:,i]!=0,data[:,j]!=0))
                k+=1
        si = np.array([ np.nansum(col[col!=0]) / np.nansum(col!=0) for col in data.T ])
    else:
        k=0
        for i in xrange(N-1):
            for j in xrange(i+1,N):
                sisj[k] = np.nansum(data[:,i]*data[:,j]*weighted)
                k+=1
        si = np.nansum(data*weighted[:,None],0)

    if concat:
        return np.concatenate((si,sisj))
    else:
        return si, sisj

def pair_corr(data,weighted=None,return_square=False,ignorenan=False,subtractmean=True):
    """
    Calculate thecorrelation matrix <sisj>-<si><sj>.

    Parameters
    ----------
    data (ndarray)
    **kwargs:
    weighted (np.ndarray,None) : 
        Calculate single and pairwise means given fractional weights for each state in the data such that a
        state only appears with some weight, typically less than one.
    return_square (bool,False)
        return Cij matrix with variances
    ignorenan (bool=False)
        If True, exclude nan from calculation. This may return a set of correlations that are incompatible
        with a joint probability distribution.
    subtractmean (bool=True)
        Subtract product of means <si><sj>.

    Returns
    -------
    cij : ndarray
    cijMat : ndarray
        Only returned if return_square is True.
    """
    (S,N) = data.shape
    cij = np.zeros(N*(N-1)//2)

    if ignorenan:
        k=0
        for i in range(N-1):
            for j in range(i+1,N):
                nanix=np.logical_or( np.isnan(data[:,i]),np.isnan(data[:,j]) )
                if subtractmean:
                    cij[k] = ( (data[nanix==0,i]*data[nanix==0,j]).mean() - 
                                data[nanix==0,i].mean()*data[nanix==0,j].mean() )
                else:
                    cij[k] = (data[nanix==0,i]*data[nanix==0,j]).mean()
                k+=1

        if return_square:
            cijMat = squareform(cij)
            cijMat[np.eye(N)==1] = si
            return cij,cijMat
        else:
            return cij
    else:
        if weighted is None:
            weighted = np.ones((data.shape[0]))
        if subtractmean:
            si = np.sum(data*np.expand_dims(weighted,1),0)/float(np.sum(weighted))
        k=0
        for i in range(N-1):
            for j in range(i+1,N):
                if subtractmean:
                    cij[k] = np.sum(data[:,i]*data[:,j]*weighted)/float(np.sum(weighted))\
                              -si[i]*si[j]
                else:
                    cij[k] = np.sum(data[:,i]*data[:,j]*weighted)/float(np.sum(weighted))
                k+=1

        if return_square:
            cijMat = squareform(cij)
            cijMat[np.eye(N)==1] = si
            return cij,cijMat
        else:
            return cij

def nan_calc_sisj(data):
    """
    Each col in data should correspond to a set votes by one spin.
    2013-12-19
    """
    (S,N) = data.shape
    sisj = np.zeros(N*(N-1)//2)

    k=0
    for i in range(N-1):
        for j in range(i+1,N):
            sisj[k] = np.nansum(data[:,i]*data[:,j])/ \
                float(np.sum(np.isnan(data[:,i]*data[:,j])==0))
            k+=1

    return ( np.nansum(data,0)/np.sum(np.isnan(data)==0,0), sisj )

def state_probs(*args,**kwargs):
    """
    Alias for get_state_probs()
    """
    return get_state_probs(*args,**kwargs)

def get_state_probs(v,allstates=None,weights=None,normalized=True):
    """
    Get probability of unique states. There is an option to allow for weights counting of the words.
    
    Params:
    -------
    states (ndarray)
        nsamples x ndim
    weights (vector)
    normalized (bool=True)
        Return probability distribution instead of frequency count
    
    Returns:
    --------
    freq (ndarray) : vector of the probabilities of each state
    """
    if v.ndim==1:
        v = v[:,None]
    n = v.shape[1]
    return_all_states = False

    if allstates is None:
        allstates = v[unique_rows(v)]
        uniqIx = unique_rows(v,return_inverse=True)
        freq = np.bincount( uniqIx )
        return_all_states = True
    else:
        if weights is None:
            weights = np.ones((v.shape[0]))
        
        freq = np.zeros(allstates.shape[0])
        for j,vote in enumerate(allstates):
            ix = ( vote==v ).sum(1)==n
            freq[j] = (ix*weights).sum()
        if np.isclose(np.sum(freq),np.sum(weights))==0:
            import warnings
            warnings.warn("States not found in given list of all states.")
    if normalized:
        freq = freq.astype(float)/np.sum(freq)

    if return_all_states:
        return freq,allstates
    return freq

def calc_sn(n,S):
    """
        Calculate nth order entropy using cluster expansion.
        2013-06-26
    """
    from itertools import combinations

    N = S[0].size
    Sn = []

    if n>=1:
        Sn.append( np.sum( S[0] ) )
    if n>=2:
        Sn.append(Sn[0])
        for index in combinations( range(N),2 ):
            i,j = index
            s = S[0][i]+S[0][j] -S[1][i,j]
#            if not np.isnan(s):
            Sn[1] -= s
#            elif n==2:
#                print 1
    if n>=3:
        Sn.append(Sn[1])
        for index in combinations( range(N),3 ):
            i,j,k = index
            s = S[0][i]+S[0][j]+S[0][k] -S[1][i,j]-S[1][j,k]-S[1][i,k] \
                    +S[2][i,j,k]
#            if not np.isnan(s):
            Sn[2] += s
    if n>=4:
        for index in combinations( range(N),4 ):
            i,j,k,l = index
            s = S[0][i]+S[0][j]+S[0][k]+S[0][l] \
                -S[1][i,j]-S[1][i,k]-S[1][i,l]-S[1][j,k]-S[1][j,l]-S[1][k,l] \
                +S[2][i,j,k]+S[2][j,k,l]+S[2][i,k,l]+S[2][i,j,l] \
                -S[3][i,j,k,l]

#    N = dS[0].size
#    S = 0
#
#    if n==2:
#        k=0
#        for i in range(N-1):
#            for j in range(i+1,N):
#                s = dS[1][k]+dS[0][i]+dS[0][j]
#                if not np.isnan(s):
#                    S += s
#                k+=1
    return Sn

def calc_ds(H):
    """
        Calculate contribution to entropy by clusters of increasing size.
        2013-06-24
    """
    n = H[0].size
    dH = []

    if n>=1:
        dH.append(H[0].copy())
    if n>=2:
        dH.append( np.zeros((n*(n-1)/2)) )
        k=0
        for i in range(n-1):
            for j in range(i+1,n):
                dH[1][k] = H[1][i,j]-dH[0][i]-dH[0][j]
                k+=1
    if n>=3:
        dH.append( np.zeros((n,n,n)))
        l=0
        for index in combinations(range(n),3):
            i,j,k = index
    return dH

def calc_s(n,p):
    """
        Function for calculating the nth order entropy given the set of
        {pi,pij,pijk,...pn}.
        2013-06-25
    """
    from itertools import combinations
    import warnings

    S = []
    n = p[0].size

    if n>=1:
        S.append( -p[0]*np.log2(p[0])-(1-p[0])*np.log2(1-p[0]) )
    if n>=2:
        k=0
        S.append(np.zeros((n,n)) )
        for index in combinations(range(p[0].size),2):
            i,j = index

#            print p[0][i]-p[1][i,j],p[0][j]-p[1][i,j],1-p[0][i]-p[0][j]+p[1][i,j]

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                S[1][i,j] = -p[1][i,j]*np.log2(p[1][i,j])- \
                    (p[0][i]-p[1][i,j])*np.log2(p[0][i]-p[1][i,j])- \
                    (p[0][j]-p[1][i,j])*np.log2(p[0][j]-p[1][i,j])- \
                    (1-p[0][i]-p[0][j]+p[1][i,j])*np.log2(1-p[0][i]-p[0][j]+p[1][i,j])

                if len(w):
                    # do something with the first warning
                    print i,j
                    print p[1][i,j],p[0][i]-p[1][i,j],p[0][j]-p[1][i,j],1-p[0][i]-p[0][j]+p[1][i,j]
    if n>=3:
        l=0
        S.append(np.zeros((n,n,n)))
        for index in combinations(range(p[0].size),3):
            i,j,k = index
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                S[2][i,j,k] = -np.nansum([p[2][i,j,k]*np.log2(p[2][i,j,k]), \
                    (p[1][i,j]-p[2][i,j,k])*np.log2(p[1][i,j]-p[2][i,j,k]), \
                    (p[1][i,k]-p[2][i,j,k])*np.log2(p[1][i,k]-p[2][i,j,k]), \
                    (p[1][j,k]-p[2][i,j,k])*np.log2(p[1][j,k]-p[2][i,j,k]), \
                    (p[0][i]-p[1][i,j]-p[1][i,k]+p[2][i,j,k])*np.log2(p[0][i]-p[1][i,j]-p[1][i,k]+p[2][i,j,k]), \
                    (p[0][j]-p[1][i,j]-p[1][j,k]+p[2][i,j,k])*np.log2(p[0][j]-p[1][i][j]-p[1][j][k]+p[2][i,j,k]), \
                    (p[0][k]-p[1][i,k]-p[1][j,k]+p[2][i,j,k])*np.log2(p[0][k]-p[1][i,k]-p[1][j,k]+p[2][i,j,k]), \
                    (1-p[0][i]-p[0][j]-p[0][k]+p[1][i,j]+p[1][i,k]+p[1][j,k]-p[2][i,j,k])*np.log2(1-p[0][i]-p[0][j]-p[0][k]+p[1][i,j]+p[1][i,k]+p[1][j,k]-p[2][i,j,k])])
            if len(w):
                print i,j,k
    return S

def calc_sijk(data,vecout=False,weighted=False):
    """
        Calculate sijk in {0,1}.
        2014-08-22
    """
    from itertools import combinations
    from scipy.special import binom
    n = data.shape[1]
    if weighted is False:
        weighted = np.ones((data.shape[0]))/data.shape[0]
    
    if vecout:
        sijk = np.zeros((binom(n,3)))
        j = 0
        for i in combinations(range(n),3):
            sijk[j] = np.sum(weighted*(data[:,i[0]]*data[:,i[1]]*data[:,i[2]]))
            j += 1
    else:
        sijk = np.zeros((n,n,n))

        for i in combinations(range(n),3):
            sijk[i[0],i[1],i[2]] = np.sum(weighted*(
                data[:,i[0]]*data[:,i[1]]*data[:,i[2]]))
    return sijk


def nan_calc_sijk(data):
    """
        Calculate sijk in {0,1}.
        2013-06-26
    """
    from itertools import combinations

    n = data.shape[1]
    sijk = np.zeros((n,n,n))
    l=0

    for i in combinations(range(n),3):
        ijk = data[:,i[0]]*data[:,i[1]]*data[:,i[2]]
        sijk[i[0],i[1],i[2]] = np.nansum(ijk)/float(np.sum(np.isnan(ijk)==0))
        l+=1

    return sijk

def nth_corr(data,n,weighted=False,exclude_empty=False,return_sample_size=False):
    """
    Compute the nth order correlations in the data.
    
    Parameters
    ----------
    data : ndarray
        n_samples x n_dims)
    n : int
        Order of correlation to compute.
    weighted : bool,False
    exclude_empty : bool,False
        Do not combine with weighted.
    return_sample_size: bool,False
        If True, return the number of samples used to calculated the correlations for each data point. This
        only works for exclude_empty.

    Returns
    -------
    Vector of nth order correlations.
    """
    from itertools import combinations
    from scipy.special import binom
    assert 1<n<=data.shape[1], "n cannot be larger than size of system in data."
    
    if weighted or not exclude_empty:
        weighted = np.ones((data.shape[0]))/data.shape[0]
        
        corr = np.zeros( int(binom(data.shape[1],n)) )
        j = 0
        for i in combinations(range(data.shape[1]),n):
            arr = data[:,i]
            corr[j] = weighted.dot( np.prod(arr,1) )
            j += 1
        return corr
    else:
        if return_sample_size:
            corr = np.zeros( int(binom(data.shape[1],n)) )
            sampleSize = np.zeros( int(binom(data.shape[1],n)) )

            j = 0
            for i in combinations(range(data.shape[1]),n):
                arr = data[:,i]
                nonzeroix = (arr!=0).all(1)
                if nonzeroix.any():
                    corr[j] = np.prod(arr[nonzeroix],1).mean(0)
                    sampleSize[j] = nonzeroix.sum()
                else:
                    corr[j] = np.nan
                j += 1

            return corr,sampleSize
        else:
            corr = np.zeros( int(binom(data.shape[1],n)) )
            j = 0
            for i in combinations(range(data.shape[1]),n):
                arr = data[:,i]
                nonzeroix = (arr!=0).all(1)
                if nonzeroix.any():
                    corr[j] = np.prod(arr[nonzeroix],1).mean(0)
                else:
                    corr[j] = np.nan
                j += 1

            return corr

def convert_sisj(si,sisj,convertTo='11',concat=True):
    """
    Convert <sisj> between 01 and 11 formulations.
    2015-06-28

    Parameters
    -------
    sisj (ndarray)
    si (ndarray)
    convertTo (str,'11')
        '11' will convert {0,1} formulation to +/-1 and '01' will convert +/-1 formulation to {0,1}
    concat : bool,True

    Returns
    -------
    si
        Converted to appropriate basis
    sisj
        converted to appropriate basis
    """
    if convertTo=='11':
        newsisj = np.zeros(sisj.shape)
        k = 0
        for i in range(len(si)-1):
            for j in range(i+1,len(si)):
                newsisj[k] = 4*sisj[k] - 2*si[i] - 2*si[j] + 1
                k += 1
        newsi = si*2-1
    else:
        newsisj = np.zeros(sisj.shape)
        k = 0
        for i in range(len(si)-1):
            for j in range(i+1,len(si)):
                newsisj[k] = ( sisj[k] + si[i] + si[j] + 1 )/4.
                k += 1
        newsi = (si+1)/2
    if concat:
        return np.concatenate((newsi,newsisj))
    return newsi,newsisj

