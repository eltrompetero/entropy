from __future__ import division
from entropy import *
from scipy.optimize import minimize,fmin
from pathos.multiprocessing import cpu_count,Pool
import numpy as np
from scipy.optimize import fmin,minimize
from numba import jit

class Bottleneck(object):
    def __init__(self,N,Nc):
        """
        Class for implementing minimization routine for Information Bottleneck soft clustering for soft votes.
        2016-04-15
        
        Subfields:
        ----------
        accuracy (float)
            Accuracy term in cost function L
        bottleneck (float)
            Bottleneck term in cost function L
        clusterAssignP (ndarray)
            n_clusters x n_spins, cluster assignment probabilties

        Methods:
        --------
        calc_Delta
        calc_P_sc
        calc_P_sc_given_si
        calc_P_sc_and_S
        chi
        setup
            Setup methods for solving.
        solve
        """
        self.N = N
        self.Nc = Nc
        self.gamma = 1  # penalty tradeoff
        self.beta = 10  # inverse temperature for soft spins
        self.hasBeenSetup = False
        self.rng = np.random.RandomState()

    def calc_P_sc(self,PofSi,Sc):
        """
        P({s_C}) for a particular {s_C} 
        Must sum over all states {s_i}
        2016-04-17
       
        Params:
        -------
        PofSi (ndarray)
            n_states vector of the probabilities of given states
        Sc (ndarray)
            n_clusters, fixed vector of cluster orientations (function that loops this function will iterate over these)
        """
        # Iterate over all clusters to get the product of the Delta_C's.
        dp = np.zeros((self.Nc,len(PofSi)))
        for i,sc in enumerate(Sc):
            dp[i,:] = self.Deltas[i,(sc==1).astype(int),:]
        return (dp*PofSi[None,:]).sum(1).prod()
        #return (self.Deltas[:,(Sc==1).astype(int),:]*PofSi[None,:]).sum(1).prod()

    def calc_P_sc_and_S(self,PofSi,Si,Sc=None):
        """
        Calculate matrix of P({s_C}|{s_i},S) of all possible states {s_C} given some data {s_i}. This means that I should iterate overall the observed states {s_i} and compute the probabilities of possible states {s_C}.

        Params:
        -------
        clusterAssignProbs (ndarray)
            n_clusters x n_spins
        PofSi (ndarray)
            n_samples, probabilities of given states (next arg)
        Si (ndarray)
            n_samples x n_spins
        Sc (ndarray=None)
            Subset of cluster votes that should be counted
        """
        # If Sc is not given, then consider all possible 2**Nc cluster states.
        if Sc is None:
            def refresh_ScGen():
                return xbin_states(self.Nc,sym=True)
            ScGen = refresh_ScGen()
            Psc = np.zeros((2,2**self.Nc))
        else:
            def refresh_ScGen():
                return iter(Sc)
            ScGen = refresh_ScGen()
            Psc = np.zeros((2,len(Sc)))

        P_Sc_and_S = np.zeros((2,2**self.Nc))
        for j,sc in enumerate(ScGen):
            negIx = Si.sum(1)<0

            dp = np.ones((negIx.sum()))
            for kc,k in enumerate(sc):
                dp *= self.Deltas[kc,int(k==1),negIx]
            dp *= PofSi[negIx]
            P_Sc_and_S[0,j] = dp.sum()
            
            dp = np.ones(((negIx==0).sum()))
            for kc,k in enumerate(sc):
                dp *= self.Deltas[kc,int(k==1),negIx==0]
            dp *= PofSi[negIx==0]
            P_Sc_and_S[1,j] = dp.sum()

        return P_Sc_and_S
 
    def calc_Delta_for_Si( self,clusterAssignPGivenC,si,sc):
        """
        Calculate Delta_C for all given microscopic states {s_i} for all given orientations s_C, the vote of C.
        
        Params:
        -------
        clusterAssignPGivenC (ndarray)
            n_clusters, P(C|i) for this particular C
        si (ndarray)
            n_samples x n_spins, microsocopic data state to consider
        sc (ndarray)
            cluster states to use
        
        Value:
        ------
        Return n_clusters x n_spin_states
        """
        assert clusterAssignPGivenC.ndim==1 and sc.ndim==1 and si.ndim==2
        return self.calc_Delta( clusterAssignPGivenC[None,:,None],sc[:,None],
                                np.swapaxes(si[:,:,None],2,0) )
    
    def define_Deltas(self,clusterAssignP,Si):
        self.Deltas = np.zeros((self.Nc,2,len(Si)))
        for i in xrange(self.Nc):
            self.Deltas[i,:,:] = self.calc_Delta_for_Si( clusterAssignP[i],Si,np.array([-1.,1.]) )

    def bottleneck_term(self,PSi,Si,Sc=None):
        """
        Calculate the first term. I(Sigma_i,Sigma_C)
        2016-04-17
        """
        # If Sc is not given, then consider all possible 2**Nc cluster states.
        if Sc is None:
            def refresh_ScGen():
                return xbin_states(self.Nc,sym=True)
            ScGen = refresh_ScGen()
            P_sc = np.zeros((2**self.Nc))
        else:
            def refresh_ScGen():
                return iter(Sc)
            ScGen = refresh_ScGen()
            P_sc = np.zeros((len(Sc)))
        
        # Iterate over all cluster orientations possible.
        for i,Sc in enumerate(ScGen):
            # Given a particular cluster orientation, sum over all data that is consistent with this.
            P_sc[i] = self.calc_P_sc( PSi,Sc )
        
        H = -np.nansum(P_sc*np.log2(P_sc))
        return H

    def accuracy_term( self,PofSi,Si,Sc ):
        """
        Information between cluster votes and final vote.
        2016-04-17
        """
        P_sc_and_S = self.calc_P_sc_and_S( PofSi,Si,Sc=Sc )
        return MI( P_sc_and_S )
    
    def setup(self,PSi,Si,Sc=None):
        """
        2016-04-15
        """
        beta = self.beta
        
        #@jit(nopython=True)
        def chi(clusterAssignP,sc,si):
            return beta*sc*(clusterAssignP*si).sum(1)
        
        #@jit(nopython=True)
        def calc_Delta(clusterAssignP,sc,si):
            return 1/(1+np.exp(-chi(clusterAssignP,sc,si)))
        
        self.chi = chi
        self.calc_Delta = calc_Delta

        def L(params,returnSeparate=False):
            if np.any(params<=0):
                return np.inf
            
            clusterAssignP = self.reshape_and_norm(params)
            # For each cluster, there are 2 possible outcomes given all {s_i}.
            self.define_Deltas(clusterAssignP,Si)
            
            # Calculate the first term. I(Sigma_i,Sigma_C)
            bottleneck = self.bottleneck_term( PSi,Si,Sc=Sc )
                
            # Calculate second term: I(Sigma_C;Sigma)
            accuracy = self.accuracy_term( PSi,Si,Sc )
            
            # Min the entropy of the code while max overlap with final vote outcome.
            if returnSeparate:
                return bottleneck, accuracy
            return self.gamma * bottleneck - accuracy
        self.L = L
        self.hasBeenSetup = True
        
    def solve(self,initialGuess=None,method='powell'):
        """
        2016-04-15
        """
        assert self.hasBeenSetup, "Must run setup() first."
        if initialGuess is None:
            initialGuess = self.rng.rand(self.Nc*self.N)
        
        if method=='fmin':
            self.soln = self.reshape_and_norm( fmin(self.L,initialGuess) )
            self.clusterAssignP = self.reshape_and_norm( self.soln )
        else: 
            self.soln = minimize(self.L, initialGuess, method=method)
            self.clusterAssignP = self.reshape_and_norm( self.soln['x'] )
        
        self.bottleneck,self.accuracy = self.L( self.clusterAssignP.ravel(),True )
        return self.clusterAssignP

    def solve_many_times(self,nIters,nJobs=None,**kwargs):
        """
        2016-04-17
        """
        if nJobs is None:
            nJobs = cpu_count()
        Solns = []
        
        def f(i):
            self.rng = np.random.RandomState()
            soln = [ self.solve(self.rng.rand(self.Nc*self.N),**kwargs) ]
            soln.append( self.L(self.clusterAssignP,True) )
            return soln
        
        if nJobs>0:
            p = Pool(nJobs)
            Solns = p.map( f,range(nIters) )
        else:
            Solns = []
            for i in xrange(nIters):
                Solns.append( f(i) )
        
        return Solns
    
    def reshape_and_norm(self,x):
        x = np.reshape(x,(self.Nc,self.N,))
        x /= x.sum(0)[None,:]
        return x

def unique_states_and_p(v):
    """
    Given set of states, return probabilities of unique states and the unique states.
    2016-04-16

    Params:
    -------
    v (ndarray)
        n_samples x n_spins
    """
    # P({s_i}), all observed individual voting configurations
    PofSi = np.bincount(unique_rows(v,return_inverse=True))
    PofSi = PofSi/PofSi.sum()
    
    # Unique {s_i}
    uniqueSi = v[unique_rows(v)]
    return PofSi,uniqueSi



def L(PofCgivenI,T,Nc,v,iprint=False):
    """
    Cost function.

    Can maximize the information that each cluster, the sum, or the set of cluster votes has individually about the final outcome.
    2016-03-25
    """
    K = v.shape[0]
    avgVote = v.sum(1)
    avgVote[np.isclose(avgVote,0)] = 0.
    avgVote = np.sign(avgVote)

    # For each cluster, compute the information it has about the average outcome.
    clustersVote = np.zeros((K,Nc))
    for i in xrange(Nc):
        clustersVote[:,i] = (PofCgivenI[:,i][None,:]*v).sum(1)
    clustersVote[np.isclose(clustersVote,0)] = 0.
    clustersVote = np.sign(clustersVote)
    
    pmat = joint_p_mat( np.vstack(( enumerate_states(clustersVote),avgVote )).T,np.array([0]),np.array([1]) )
    info = MI(pmat)
    cost = Snaive(clustersVote)
    
    if iprint:
        print "info %f" %info
        print "cost %f"%cost
    return info - T*cost

def wrap_L(params,T,Nc,v):
    N = v.shape[1]
    if np.any(np.array(params)<0):
        return 1e30
    PofCgivenI = params.reshape(N,Nc)
    PofCgivenI = PofCgivenI / PofCgivenI.sum(1)[:,None]
    return -L(PofCgivenI,T,Nc,v)

def solve(T,Nc,v,maxTries=100,method='minimize',rng=np.random.RandomState()):
    """
    many different starting conditions.
    """
    N = v.shape[1]
    i = 0
    soln = {'success':False}
    while (not soln['success']) and i<maxTries:
        soln = minimize( wrap_L,rng.rand(N*Nc),args=(T,Nc,v) )
        i += 1
    if soln['success']:
        soln['x'] = soln['x'].reshape(N,Nc)
        # Normalize probabilities.
        soln['x'] = soln['x'] / soln['x'].sum(1)[:,None]
        return soln
    return 0

def solve_parallel(nSolns,T,Nc,v,nJobs=None,**kwargs):
    """
    Run solve() in parallel
    """
    if nJobs is None:
        nJobs = cpu_count()
    assert nJobs>0
   
    def wrapped_solve(i):
        rng = np.random.RandomState()
        return solve(T,Nc,v,rng=rng,**kwargs)

    p = Pool(nJobs)
    Solutions = p.map(wrapped_solve,range(nSolns))
    p.close()
    return Solutions

def get_best(Solutions,n):
    """
    Get best solutions as given by the energy function you're trying to minimize. This is useful when there are a lot of solutions with many bad ones.
    """
    ix = np.argsort([s['fun'] for s in Solutions])
    return [Solutions[i] for i in ix[:n]]

def hard_clusters(clusterIx,v):
    """
    Put voters into clusters and return majority votes of those clusters.
    2016-03-29
    """
    n_clusters = len(np.unique(clusterIx))
    clustersVote = np.zeros((v.shape[0],n_clusters))
    
    for i,ix in enumerate(np.unique(clusterIx)):
        clustersVote[:,i] = v[:,np.where(clusterIx==ix)[0]].sum(1)
    clustersVote[np.isclose(clustersVote,0)] = 0
    clustersVote = np.sign(clustersVote)
    return clustersVote
