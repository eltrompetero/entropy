from __future__ import division
from entropy import joint_p_mat,MI,bin_states,calc_cij,joint_p_mat,enumerate_states,Snaive
from scipy.optimize import minimize,fmin
from pathos.multiprocessing import cpu_count,Pool
import numpy as np

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
    Put voters into most likely clusters and return majority votes of those clusters.
    2016-03-29
    """
    n_clusters = len(np.unique(clusterIx))
    clustersVote = np.zeros((v.shape[0],n_clusters))
    
    for i,ix in enumerate(np.unique(clusterIx)):
        clustersVote[:,i] = v[:,np.where(clusterIx==ix)[0]].sum(1)
    clustersVote[np.isclose(clustersVote,0)] = 0
    clustersVote = np.sign(clustersVote)
    return clustersVote
