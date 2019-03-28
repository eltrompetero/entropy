# ===================================================================================== #
# Module for entropy estimators.
# Author: Eddie Lee, edlee@alumni.princeton.edu
# ===================================================================================== #
import numpy as np
from .entropy import *
from warnings import warn
from misc.utils import unique_rows
import matplotlib.pyplot as plt


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
    Calculate Ma bound on entropy by partitioning the data set according to the number of
    votes in the majority.  See Strong et al. 1998 and Lee et al. 2015.

    This is a simply application of the convexity of the entropy function.

    Parameters
    ----------
    data : ndarray

    Returns
    -------
    float
        Ma estimate in bits.
    """
    
    if data.ndim==1:
        data = data[:,None]
    else:
        assert data.ndim==2

    _,p = np.unique(data, axis=0, return_counts=True)
    p = p/p.sum()
    
    return -np.log2(p.dot(p))

def S_quad(X, sample_fraction, n_boot,
           X_is_count=False,
           return_fit=False,
           rng=None,
           parallel=False,
           symmetrize=False,
           fit_order=2,
           fit_tol=.01,
           disp=False):
    """
    Calculate entropy using quadratic extrapolation to infinite data size. The points used
    to make the extrapolation are given in sample_fraction. Units of bits.

    Flexibility to do a spline fit of any order.

    Parameters
    ----------
    X : ndarray
        Either a list of indices specifying states or a list of the number of times each
        state occurs labeled by the array index.
    sample_fraction : ndarray
        Fraction of sample to use to extrapolate entropy to infinite sample size.
    n_boot : int
        Number of times to bootstrap sample to estimate mean entropy at each sample size.
    X_is_count : bool, False
        If X is not a sample but a count of each state in the data.
    return_fit : bool, False
        If True, return statistics of quadratic fit.
    rng : np.random.RandomState, None
    parallel : bool, False
    symmetrize : bool, False
        If True, all probabilities are halved and then measurements are doubled.
    fit_order : int, 2
        If 0, then order is increased til the change in the estimate is below fit_tol.
    fit_tol : float, 0.01

    Returns
    -------
    float
        Estimated entropy in bits.
    ndarray, optional
        Coefficients for quadratic fit.
    float, optional
        Fit error.
    mpl.plt.Figure, optional
        Visual rep. of fit.
    """
    
    assert X.ndim==1
    if not type(sample_fraction) is np.ndarray:
        sample_fraction = np.array(sample_fraction)

    rng = rng or np.random
    
    if not X_is_count:
        X = np.unique(X, return_counts=True)[1]
    Xsum = X.sum()
    assert Xsum>1, "Must provide more than one state to calculate bootstrap."

    if parallel:
        estS = _S_quad_parallel(X, sample_fraction, n_boot, symmetrize)
    else:
        estS = np.zeros((len(sample_fraction),n_boot))
        ix = list(range(len(X)))
        pState = X/Xsum

        for i,f in enumerate(sample_fraction):
            for j in range(n_boot):
                bootSample = rng.choice(ix, size=int(f*Xsum), p=pState)
                p = np.unique(bootSample, return_counts=True)[1] / bootSample.size
                if symmetrize:
                    p = np.concatenate((p/2,p/2))
                estS[i,j] = -p.dot(np.log2(p))
    
    if fit_order:
        fit = np.polyfit(1/np.floor(sample_fraction*Xsum), estS.mean(1), fit_order)
    else:
        # find lowest order of fit that works well
        x, y = 1/np.floor(sample_fraction*Xsum), estS.mean(1)
        fit_order = 1
        fit = np.zeros(1)
        
        close = False
        while fit_order<len(sample_fraction) and not close:
            newfit = np.polyfit(x, y, fit_order+1)
            if abs(fit[-1]-newfit[-1])<fit_tol:
                close = True
            else:
                fit = newfit
                fit_order += 1
        if fit_order==len(sample_fraction):
            warn("Optimal fit order is large.")

    err = np.polyval(fit, 1/np.floor(sample_fraction*Xsum)) - estS.mean(1)
    if fit[0]>0:
        print("Fit curvature is positive.")
    
    if disp:
        fig, ax = plt.subplots()
        ax.plot(1/np.floor(sample_fraction*Xsum), estS.mean(1), '.')

        x = np.linspace(0, 1/np.floor(sample_fraction.min()*Xsum))
        ax.plot(x, np.polyval(fit, x), 'k-')
    
    output = [fit[-1]]
    if return_fit:
        output += [fit, err]
    if disp:
        output.append(fig)

    if len(output)==1:
        return output[0]
    return tuple(output)

def _S_quad_parallel(X, sample_fraction, n_boot,
                     symmetrize):
    """
    Parameters
    ----------
    X : ndarray
    sample_fraction : ndarray
    n_boot : int
    symmetrize : bool

    Returns
    -------
    ndarray
    """
    
    from multiprocess import Pool, cpu_count
    estS = np.zeros((len(sample_fraction),n_boot))
   
    ix = list(range(len(X)))
    Xsum = X.sum()
    assert Xsum>1, "Must provide more than one state to calculate bootstrap."
    pState = X/Xsum
    
    def g(args):
        f, rng = args
        estS = np.zeros(n_boot)
        for j in range(n_boot):
            bootSample = rng.choice(ix, size=int(f*Xsum), p=pState)
            p = np.unique(bootSample, return_counts=True)[1] / bootSample.size
            if symmetrize:
                p = np.concatenate((p/2,p/2))
            estS[j] = -(p*np.log2(p)).sum()
        return estS

    pool = Pool(cpu_count()-1)
    estS = np.vstack(pool.map( g, [(f,np.random.RandomState()) for f in sample_fraction] ))
    pool.close()
    
    return estS

def S_naive(samples):
    """
    Calculate entropy in bits.

    Parameters
    ----------
    samples : ndarray
        With 2 dim. Each row is a sample.

    Returns
    -------
    float 
        Estimate of entropy in bits.
    """

    if samples.ndim==1:
        samples = samples[:,None]
    else:
        assert samples.ndim==2

    _, freq = np.unique(samples, axis=0, return_counts=True)
    p = freq/freq.sum()
    return -np.sum(p*np.log2(p))

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
