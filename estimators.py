# ===================================================================================== #
# Module for entropy estimators.
# Author: Eddie Lee, edlee@alumni.princeton.edu
# ===================================================================================== #
import numpy as np
from warnings import warn
from multiprocess import Pool, cpu_count
import matplotlib.pyplot as plt
from itertools import combinations

from .entropy import *



# ======= #
# Methods #
# ======= #
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
        digitizedStates = np.unique(XY, axis=0, return_inverse=True)
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

    _, p = np.unique(data, axis=0, return_counts=True)
    p = p/p.sum()
    
    return -np.log2(p.dot(p))

def S_poly(X, sample_fraction, n_boot,
           X_is_count=False,
           return_fit=False,
           rng=None,
           parallel=False,
           symmetrize=False,
           fit_order=2,
           fit_tol=.01,
           disp=False):
    """Calculate entropy using polynomial extrapolation to infinite data size. The points
    used to make the extrapolation are given in sample_fraction. Units of bits.

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
        If True, return statistics of polynomial fit.
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
        Coefficients for polynomial fit.
    float, optional
        Fit error.
    mpl.plt.Figure, optional
        Visual rep. of fit.
    """
    
    # check inputs
    assert X.ndim==1
    if not type(sample_fraction) is np.ndarray:
        if hasattr(sample_fraction,'__len__'):
            sample_fraction = np.array(sample_fraction)
        else:
            sample_fraction = np.array([sample_fraction])
    rng = rng or np.random
    if not X_is_count:
        X = np.unique(X, return_counts=True)[1]
    Xsum = X.sum()
    assert Xsum>1, "Must provide more than one state to calculate bootstrap."
    
    # Generate bootstrap sample for all fractions in sample_fraction
    if parallel:
        estS = _S_poly_parallel(X, sample_fraction, n_boot, symmetrize)
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
    
    # fit polynomial to averages of bootstrapped samples
    if fit_order:
        fit = np.polyfit(1/np.floor(sample_fraction*Xsum), estS.mean(1), fit_order)
    else:
        # find lowest order of fit that works well
        x, y = 1/np.floor(sample_fraction*Xsum), estS.mean(1)
        fit_order = 1
        fit = np.zeros(1)
        
        close = False
        while fit_order<(len(sample_fraction)-1) and not close:
            newfit = np.polyfit(x, y, fit_order+1)
            if abs(fit[-1]-newfit[-1])<fit_tol:
                close = True
            else:
                fit = newfit
                fit_order += 1
        if fit_order==(len(sample_fraction)-1):
            warn("Optimal fit order is large.")

    err = np.polyval(fit, 1/np.floor(sample_fraction*Xsum)) - estS.mean(1)
    if fit[0]>0:
        print("Fit curvature is positive.")
    
    # optional plotting
    if disp:
        fig, ax = plt.subplots()
        ax.plot(1/np.floor(sample_fraction*Xsum), estS.mean(1), '.')

        x = np.linspace(0, 1/np.floor(sample_fraction.min()*Xsum))
        ax.plot(x, np.polyval(fit, x), 'k-')
    
    # output
    output = [fit[-1]]
    if return_fit:
        output += [fit, err]
    if disp:
        output.append(fig)

    if len(output)==1:
        return output[0]
    return tuple(output)

def _S_poly_parallel(X, sample_fraction, n_boot,
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

# ======= #
# Classes # 
# ======= #
class ClusterEntropy():
    def __init__(self, X):
        """Class for using idea in Cocco and Monasson (2011) to iteratively estimate
        the entropy of a pairwise maxent model on binary spins.

        In order to determine an estimate, it is important to define a threshold
        theta at which cluster estimates are excluded from the sum.

        Built to harness multiple threads to iterate over sets of clusters.

        Parameters
        ----------
        X : ndarray
        """
        
        self.X = X
        
        # keep track of all cluster entropies calculated
        self.dS = {}
        self.S = {}

    def calc_S(self, clus, cache=True):
        """Calculate entropy of given cluster and cache it into dict of entropy S.

        Parameters
        ----------
        clus : set
            List of spin indices to include in this cluster.

        Returns
        -------
        float
            Entropy in nats.
        """
        
        # no empty clusters allowed
        assert len(clus)
        if clus in self.S.keys():
            return self.S[clus]

        # naive entropy estimator
        counts = np.unique(self.X[:,list(clus)], axis=0, return_counts=True)[1]
        p = counts / counts.sum()
        S = -(p * np.log(p)).sum()
        if cache: self.S[clus] = S
        return S

    def calc_dS(self, clus, cache=True):
        """Calculate cluster entropy dS for a given cluster recursively. Adds newly
        calculated clusters to dS for use with future calculation.

        Parameters
        ----------
        clus : set
            List of spin indices to include in this cluster.

        Returns
        -------
        float
        """

        # no empty clusters allowed
        assert(len(clus))
        clus = frozenset(clus)

        # if already calculated
        if clus in self.dS.keys():
            return self.dS[clus]
        elif len(clus)==1:
            if cache: self.dS[clus] = self.calc_S(clus)
            return self.calc_S(clus, cache=cache)

        thisdS = self.calc_S(clus, cache=cache)
        for i in range(1, len(clus)):
            for subclus in combinations(clus, i):
                thisdS -= self.calc_dS(subclus, cache=cache)
        if cache: self.dS[clus] = thisdS
        return thisdS
    
    def _ordered_dS(self):
        """Set up a cluster size ordered list of dS."""
        
        if 'ordered_dS' in self.__dict__.keys():
            return self.ordered_dS
        
        ordered_dS = []
        for i in range(1, len(self.dS)+1):
            ordered_dS.append([])
            for ijk in combinations(range(self.X.shape[1]), i):
                ordered_dS[-1].append(self.calc_dS(ijk))
        self.ordered_dS = ordered_dS

    def add_clusters(self, k):
        """Generate dS for specific cluster size. See .setup_clusters().

        Parameters
        ----------
        k : int
            Cluster size to compute.
        """
        
        from scipy.special import binom
        assert k>=1

        # for all clusters of sizes k and through each cluster...
        self.ordered_dS = []
        if k==1:
            self.ordered_dS.append([])
            for ijk in combinations(range(self.X.shape[1]), k):
                self.ordered_dS[-1].append(self.calc_dS(ijk))
        else:
            n_cpus = cpu_count()
            with Pool() as pool:       
                self.ordered_dS.append([])
                def loop_wrapper(args):
                    istart, iend = args
                    counter = 0
                    for ijk_ in combinations(range(self.X.shape[1]), k):
                        if istart <= counter < iend:
                            self.calc_dS(ijk_)
                        counter += 1
                    return self.S, self.dS

                diter = binom(self.X.shape[1], k) // (n_cpus-1)
                output = list(pool.map(loop_wrapper,
                                       [(pid*diter, (pid+1)*diter) for pid in range(n_cpus)]))
                for out in output:
                    self.S.update(out[0])
                    self.dS.update(out[1])
                    self.ordered_dS[-1].extend(out[1].values())
 
    def setup_clusters(self, mx, n_cpus=None):
        """Generate dS for all clusters up to the specified max size.

        Parameters
        ----------
        mx : int
            Max cluster size to compute.
        n_cpus : int
        """
        
        from scipy.special import binom
        assert mx>=1
        n_cpus = n_cpus or cpu_count()

        # for all clusters of sizes i and through each cluster...
        self.ordered_dS = []
        for i in range(1, min(2, mx+1)):
            self.ordered_dS.append([])
            for ijk in combinations(range(self.X.shape[1]), i):
                self.ordered_dS[-1].append(self.calc_dS(ijk))

        with Pool() as pool:       
            for i in range(2, mx+1):
                self.ordered_dS.append([])
                def loop_wrapper(args):
                    istart, iend = args
                    counter = 0
                    for ijk_ in combinations(range(self.X.shape[1]), i):
                        if istart <= counter < iend:
                            self.calc_dS(ijk_)
                        counter += 1
                    return self.S, self.dS

                diter = binom(self.X.shape[1], i) // (n_cpus-1)
                output = list(pool.map(loop_wrapper,
                                       [(pid*diter, (pid+1)*diter) for pid in range(n_cpus)]))
                for out in output:
                    self.S.update(out[0])
                    self.dS.update(out[1])
                    self.ordered_dS[-1].extend(out[1].values())
        
        # multiprocessing (should be more clever but doesn't work, pipe breaks)
        #for i in range(2, mx+1):
        #    def loop_wrapper(*ijk):
        #        for ijk_ in ijk:
        #            self.calc_dS(ijk_)

        #    with Manager() as manager:
        #        # implicitly rely on shared dicts
        #        self.dS = manager.dict(self.dS)
        #        self.S = manager.dict(self.S)
        #        
        #        combos = list(combinations(range(self.X.shape[1]), i))
        #        diter = len(combos)//(n_cpus-1)
        #        for pid in range(n_cpus):
        #            p = Process(target=loop_wrapper,
        #                        args=combos[pid*diter:(pid+1)*diter])
        #            p.start()
        #        for pid in range(n_cpus):
        #            p.join()
        #        self.dS = self.dS.copy()
        #        self.S = self.S.copy()
    
    def plot_dS(self, fig=None, ax=None):
        """Show distribution of dS for each cluster size to get a sense of how well
        approximation will work.

        Parameters
        ----------
        fig : plt.Figure, None
        ax : plt.Axes, None

        Returns
        -------
        plt.Figure
        """
        
        if fig is None and ax is None:
            fig, ax = plt.subplots()
        elif not fig is None and ax is None:
            ax = fig.add_subplot(111)
        
        ordered_dS = self._ordered_dS()
        for i in range(len(ordered_dS)):
            ax.plot(np.random.normal(size=len(ordered_dS[i]), scale=.05, loc=i+1),
                    ordered_dS[i],
                    '.', c='C0', alpha=.3, mew=0)
        ax.set(xlabel='order', ylabel=r'$\Delta S$', xticks=range(1, len(ordered_dS)+1))
        ax.grid()
        
        return fig
#end ClusterEntropy
