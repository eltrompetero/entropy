# Module for entropy estimators.
# Author: Eddie Lee, edlee@alumni.princeton.edu
import numpy as np
from .entropy import *
from warnings import warn
from misc.utils import unique_rows


def cross_entropy(X, Y, method='naive', return_p=False):
    """Estimate cross entropy from two samples.
    \sum_i  p(x_i) \log_2 (p(x_i)/q(y_i))

    Parameters
    ----------
    X : ndarray
        Each row is a sample.
    Y : ndarray
        Each row is a sample.
    method : str, 'naive'
    return_p : bool,False
        If True, return estimated probability distributions for all unique states.

    Returns
    -------
    float
        Estimate of cross entropy in bits.
    duple of ndarrays
        Estimated probability distributions of X and Y. 
    """
    
    assert X.shape[1]==Y.shape[1]

    if method=='naive':
        XY = np.vstack((X,Y))
        digitizedStates = unique_rows(XY, return_inverse=True)
        uniqx = digitizedStates[:len(X)]
        uniqy = digitizedStates[len(X):]
        uniqStates = np.unique(digitizedStates)
        
        # estimate probability distributions
        px = np.zeros(len(uniqStates))
        py = np.zeros(len(uniqStates))
        for i,ix in enumerate(uniqStates): 
            px[i] = (uniqx==ix).sum()
            py[i] = (uniqy==ix).sum()

        px /= px.sum()
        py /= py.sum()
        
        if (py==0).any():
            if return_p:
                return np.nan, (px,py)
            return np.nan
        if return_p:
            return (px*np.log2(px/py)).sum(), (px,py)
        return (px*np.log2(px/py)).sum()
    else:
        raise NotImplementedError

def S_ma(data):
    """
        Calculate Ma bound on entropy by partitioning the data set according to
        the number of votes in the majority.
        See Strong et al. 1998.
    2014-02-19
    """
    from misc_fcns import unique_rows

    majCounts = np.sum(data,1)
    states = data[unique_rows(data),:]
    l = states.shape[0]
    pC = np.zeros((l))
    pK = np.zeros((l))

    i = 0
    for k in range(l):
        ix = majCounts==k
        if np.sum(ix)>0:
            pK[i] = np.sum(ix).astype(float)/ix.size

            probs = get_state_probs(data[ix,:],allstates=states)
            pC[i] = np.sum(probs**2)
        else:
            pK[i],pC[i] = 0., 0.
        i += 1
    return -np.nansum( pK*np.log2(pK*pC) )

def S_quad(X, sample_fraction, n_boot,
           X_is_count=False,
           return_fit=False,
           rng=None):
    """
    Calculate entropy using quadratic extrapolation to infinite data size. The points used to make the
    extrapoation are given in sample_fraction.

    Parameters
    ----------
    X : ndarray
        Either a list of indices specifying states or a list of the number of times each state occurs labeled
        by the array index.
    sample_fraction : ndarray
        Fraction of sample to use to extrapolate entropy to infinite sample size.
    n_boot : int
        Number of times to bootstrap sample to estimate mean entropy at each sample size.
    X_is_count : bool, False
        If X is not a sample but a count of each state in the data.
    return_fit : bool, False
        If True, return statistics of quadratic fit.
    rng : np.random.RandomState, None

    Returns
    -------
    float
        Estimated entropy in bits.
    ndarray, optional
        Coefficients for quadratic fit.
    float, optional
        Fit error.
    """
    
    rng = rng or np.random
    estS = np.zeros((len(sample_fraction),n_boot))
    ix = list(range(len(X)))
    
    if X_is_count:
        assert X.sum()>1, "Must provide more than one state to calculate bootstrap."
        pState = X/X.sum()

        for i,f in enumerate(sample_fraction):
            for j in range(n_boot):
                bootSample = rng.choice(ix, size=int(f*X.sum()), p=pState)
                p = np.unique(bootSample, return_counts=True)[1] / int(f*X.sum())
                estS[i,j] = -(p*np.log2(p)).sum()

        fit = np.polyfit(1/np.around(sample_fraction*X.sum()), estS.mean(1),2)
        err = np.polyval(fit, 1/np.around(sample_fraction*X.sum())) - estS.mean(1)
    else:
        for i,f in enumerate(sample_fraction):
            for j in range(n_boot):
                bootSample = rng.choice(ix,size=int(f*len(X)))
                p = np.unique(bootSample, return_counts=True)[1] / int(f*X.sum())
                estS[i,j] = -(p*np.log2(p)).sum()
    
        fit = np.polyfit(1/np.around(sample_fraction*len(X)), estS.mean(1), 2)
        err = np.polyval(fit, 1/np.around(sample_fraction*X.sum())) - estS.mean(1)

    if return_fit:
        return np.polyval(fit,0), fit, err
    return np.polyval(fit,0)

def S_naive(samples):
    """
    Calculate entropy in bits.
    2014-02-19
    """
    freq = get_state_probs(samples)
    return -np.sum(freq*np.log2(freq))

def fractional_dkl_quad(X,sample_fraction,n_boot,q,X_is_count=False,return_estF=False):
    """
    Calculate fraction DKL captured for binary spin system using quadratic extrapolation to infinite data size.

    Parameters
    ----------
    X : ndarray
    sample_fraction : ndarray
        Fraction of sample to use to extrapolate entropy to infinite sample size.
    n_boot : int
        Number of times to bootstrap sample to estimate mean entropy at each sample size.
    q : ndarray
        Probability distribution to compare sample with.
    X_is_count : bool,False
        If X is not a sample but a count of each state in the data.
    return_estF : bool,False
        Return the fractional DKL from bootstrap samples.

    Returns
    -------
    estF : float
    estF : ndarray
        (Optional) Estimated fractional DKL captured for bootstrap samples. (n_fraction,n_boot).
    """
    estF = np.zeros((len(sample_fraction),n_boot))
    ix = list(range(len(X)))
    
    if X_is_count:
        pState = X/X.sum()

        for i,f in enumerate(sample_fraction):
            for j in range(n_boot):
                bootSample = np.random.choice(ix,size=int(f*X.sum()),p=pState)
                p = state_probs( bootSample[:,None],allstates=np.array(ix) )
                estF[i,j] = 2-(p*np.log2(q)).sum()/np.nansum(p*np.log2(p))
        
        rowMean = estF.mean(1)
        if np.isinf(rowMean).any():
            warn("Zero entropy, ignoring.")
            rowMean = rowMean[np.isinf(rowMean)==0]
        fit = np.polyfit( 1/np.around(sample_fraction*X.sum()),rowMean,2 )

        # Simple check to make sure that the extrapolation is increasing as it approaches zero.
        #return estF,fit
        assert np.polyval(fit,0)>np.polyval(fit,1/X.sum())
    else:
        for i,f in enumerate(sample_fraction):
            for j in range(n_boot):
                bootSample = np.random.choice(ix,size=int(f*len(X)))
                p = state_probs(bootSample[:,None])[0]
                estF[i,j] = 2-(p*np.log2(q)).sum()/np.nansum(p*np.log2(p))

        rowMean = estF.mean(1)
        if np.isinf(rowMean).any():
            warn("Zero entropy, ignoring.")
            rowMean = rowMean[np.isinf(rowMean)==0]
        fit = np.polyfit( 1/np.around(sample_fraction*len(X)),rowMean,2 )

        assert np.polyval(fit,0)>np.polyval(fit,1/len(X))
    
    if np.polyval(fit,0)>1:
        warn("Extrapolated fractional DKL exceeds 1.")
    if return_estF:
        return np.polyval(fit,0),estF
    return np.polyval(fit,0)
