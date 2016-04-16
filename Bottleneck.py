from __future__ import division
from entropy import joint_p_mat,MI,bin_states,calc_cij,joint_p_mat,enumerate_states,Snaive,xbin_states
from scipy.optimize import minimize,fmin
from pathos.multiprocessing import cpu_count,Pool
import numpy as np
from scipy.optimize import fmin,minimize
from numba import jit

class Bottleneck(object):
    def __init__(self,N,Nc):
        """
        2016-04-15
        
        Subfields:
        ----------
        
        Methods:
        --------
        chi
        calc_Delta
        setup
            Setup methods for solving.
        solve

        """
        self.N = N
        self.Nc = Nc
        self.gamma = 1  # penalty tradeoff
        self.beta = 10  # inverse temperature for soft spins
        self.hasBeenSetup = False

    def calc_P_sc(self,clusterAssignP,PofSi,Si,Sc):
        """
        P({s_C}) for a particular {s_C} 
        Must sum over all states {s_i}
        2016-04-15 
       
        Params:
        -------
        PofthisCgiveni (ndarray)
            n_clusters x n_spins, P(C|i)
        PofSi (ndarray)
            n_states vector of the probabilities of given states
        Si (ndarray)
            n_states x n_spins, data to consider
        Sc (ndarray)
            n_clusters, fixed vector of cluster orientations (function that loops this function will iterate over these)
        """
        P_sc = 1
        # Iterate over all clusters to get the product of the Delta_C's.
        for j,sc in enumerate(Sc):
            innerTerm = 0
            # Iterate over all the microscopic states.
            for i,si in enumerate(Si):
                innerTerm += self.calc_Delta( clusterAssignP[j],sc,si ) * PofSi[i]
            P_sc *= innerTerm
        return P_sc
        
    def calc_P_sc_given_si( self,clusterAssignP,si,sc):
        """
        P({s_C})
        Must sum over all states {s_i}
        
        Params:
        -------
        PofthisCgiveni (ndarray)
            n_clusters x n_spins, P(C|i)
        si (ndarray)
            n_spins, data to consider
        Sc (ndarray)
            n_clusters, fixed vector of cluster orientations (function that loops this function will iterate over these)
        """
        Delta = 1.
        # Iterate over all the states.
        for j in xrange(len(sc)):
            Delta *= self.calc_Delta( clusterAssignP[j],sc[j],si )
        return Delta

        
    def calc_P_sc_and_S(self,clusterAssignProbs,PofSi,Si,Sc=None):
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

        for i,si in enumerate(Si):
            # Compute probabilities of all cluster votes given a particular microscopic config {s_i}.
            for j,sc in enumerate(ScGen):
                dp = self.calc_P_sc_given_si( clusterAssignProbs,si,sc )*PofSi[i]
                if si.sum()>0:
                    Psc[1,j] += dp
                else:
                    Psc[0,j] += dp

            ScGen = refresh_ScGen()
        P_sc_S = Psc
        return P_sc_S
        
    def bottleneck_term(self,clusterAssignP,PSi,Si,Sc):
        # Calculate the first term. I(Sigma_i,Sigma_C)
        P_sc = np.zeros((2**self.Nc))
        # Iterate over all cluster orientations possible.
        for i,Sc in enumerate(xbin_states(self.Nc,sym=True)):
            # Given a particular cluster orientation, sum over all data that is consistent with this.
            P_sc[i] = self.calc_P_sc( clusterAssignP,PSi,Si,Sc )
        
        H = -np.nansum(P_sc*np.log2(P_sc))
        return H

    def accuracy_term( self,clusterAssignP,PofSi,Si ):
        """
        Information between cluster votes and final vote.
        """
        P_sc_and_S = self.calc_P_sc_and_S( clusterAssignP,PofSi,Si )
        return MI( P_sc_and_S )
    
    def setup(self,PSi,Si,Sc):
        """
        2016-04-15
        """
        beta = self.beta
        
        @jit(nopython=True)
        def chi(clusterAssignP,sc,si):
            return beta*sc*(clusterAssignP*si).sum()
        
        @jit(nopython=True)
        def calc_Delta(clusterAssignP,sc,si):
            return 1/(1+np.exp(-chi(clusterAssignP,sc,si)))
        
        self.chi = chi
        self.calc_Delta = calc_Delta
        
        def L(params,returnSeparate=False):
            if np.any(params<=0):
                return np.inf
            
            clusterAssignP = self.reshape_and_norm(params)
            
            # Calculate the first term. I(Sigma_i,Sigma_C)
            bottleneck = self.bottleneck_term( clusterAssignP,PSi,Si,Sc )
                
            # Calculate second term: I(Sigma_C;Sigma)
            accuracy = self.accuracy_term( clusterAssignP,PSi,Si )
            
            # Min the entropy of the code while max overlap with final vote outcome.
            if returnSeparate:
                return bottleneck, accuracy
            return self.gamma * bottleneck - accuracy
        self.L = L
        self.hasBeenSetup = True
        
    def solve(self,method='fmin'):
        """
        2016-04-15
        """
        assert self.hasBeenSetup, "Must run setup() first."

        if method=='fmin':
            self.soln = self.reshape_and_norm( fmin(self.L,np.random.rand(self.Nc*self.N)) )
            self.clusterAssignP = self.reshape_and_norm( self.soln )
        else: 
            self.soln = minimize(self.L, np.random.rand(self.Nc*self.N),
                                 method=method)
            print self.soln['message']
            print self.soln['fun']
            self.clusterAssignP = self.reshape_and_norm( self.soln['x'] )
        
        self.bottleneck,self.accuracy = self.L( self.clusterAssignP.ravel(),True )
        return self.clusterAssignP
    
    def reshape_and_norm(self,x):
        x = np.reshape(x,(self.Nc,self.N,))
        x /= x.sum(0)[None,:]
        return x




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
