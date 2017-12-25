# Module for entropy estimators.
import numpy as np
from entropy import *
from warnings import warn

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

def S_quad(X,sample_fraction,n_boot,X_is_count=False):
    """
    Calculate entropy for binary spin system using quadratic extrapolation to infinite data size.

    Parameters
    ----------
    X : ndarray
    sample_fraction : ndarray
        Fraction of sample to use to extrapolate entropy to infinite sample size.
    n_boot : int
        Number of times to bootstrap sample to estimate mean entropy at each sample size.
    X_is_count : bool
        If X is not a sample but a count of each state in the data.
    """
    estS = np.zeros((len(sample_fraction),n_boot))
    ix = range(len(X))
    
    if X_is_count:
        pState = X/X.sum()

        for i,f in enumerate(sample_fraction):
            for j in xrange(n_boot):
                bootSample = np.random.choice(ix,size=int(f*X.sum()),p=pState)
                p = state_probs(bootSample[:,None])[0]
                estS[i,j] = -(p*np.log2(p)).sum()

        fit = np.polyfit(1/np.around(sample_fraction*X.sum()),estS.mean(1),2)
    else:
        for i,f in enumerate(sample_fraction):
            for j in xrange(n_boot):
                bootSample = np.random.choice(ix,size=int(f*len(X)))
                p = state_probs(bootSample[:,None])[0]
                estS[i,j] = -(p*np.log2(p)).sum()
    
        fit = np.polyfit(1/np.around(sample_fraction*len(X)),estS.mean(1),2)
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
    ix = range(len(X))
    
    if X_is_count:
        pState = X/X.sum()

        for i,f in enumerate(sample_fraction):
            for j in xrange(n_boot):
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
            for j in xrange(n_boot):
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
