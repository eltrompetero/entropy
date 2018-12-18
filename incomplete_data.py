# Module for extracting information from data sets or checking them.
# Edward D. Lee, edl56@cornell.edu
# 2017-09-28

import numpy as np
from .entropy import bin_states

# ============================================================================================== #
# Missing data.
# Functions for checking the validity of triplet probability distribution in data sets where we're
# missing data.
# ============================================================================================== #
def _bounds_case_p123(X):
    """
    When we want to bound p123.
    
    Parameters
    ----------
    X : ndarray
        n_sample,3 in terms of {-1,1} and empty data points are labeled with 0
        
    Returns
    -------
    mn,mx : floats
        Bounds on p123.
    """
    # Get all probabilities except for p123 which we are trying to bound.
    p12 = (X[:,:2]==1).all(1).sum() / X[:,:2].all(1).sum()
    p13 = (X[:,[0,2]]==1).all(1).sum() / X[:,[0,2]].all(1).sum()
    p23 = (X[:,1:]==1).all(1).sum() / X[:,1:].all(1).sum()
    p1 = (X[:,0]==1).sum() / (X[:,0]!=0).sum()
    p2 = (X[:,1]==1).sum() / (X[:,1]!=0).sum()
    p3 = (X[:,2]==1).sum() / (X[:,2]!=0).sum()
    
    # Check bounds on p123 by iterating through all the conditions and checking 
    # that the value of p123 doesn't exceed the bounds.
    mn,mx = 0,1
    mn,mx = check_bound(mn,mx,p12-1,p12)
    mn,mx = check_bound(mn,mx,p13-1,p13)
    mn,mx = check_bound(mn,mx,p23-1,p23)
    mn,mx = check_bound(mn,mx,-p1+p12+p13,1-p1+p12+p13)
    mn,mx = check_bound(mn,mx,-p2+p23+p12,1-p2+p23+p12)
    mn,mx = check_bound(mn,mx,-p3+p13+p23,1-p3+p13+p23)
    mn,mx = check_bound(mn,mx,p12+p13+p23-p1-p2-p3,1+p12+p13+p23-p1-p2-p3)
    
    return mn,mx

def _bounds_case_p23(X):
    """
    When we want to bound p23 which implies that we don't know p123.
    
    Parameters
    ----------
    X : ndarray
        n_sample,3 in terms of {-1,1} and empty data points are labeled with 0
        
    Returns
    -------
    mn,mx : floats
        Bounds on p123.
    """
    
    if type(X) is np.ndarray:
        # Get all probabilities except for p123 which we are trying to bound.
        p12 = (X[:,:2]==1).all(1).sum() / X[:,:2].all(1).sum()
        p13 = (X[:,[0,2]]==1).all(1).sum() / X[:,[0,2]].all(1).sum()
        p1 = (X[:,0]==1).sum() / (X[:,0]!=0).sum()
        p2 = (X[:,1]==1).sum() / (X[:,1]!=0).sum()
        p3 = (X[:,2]==1).sum() / (X[:,2]!=0).sum()
    elif type(X) is tuple:
        p1, p2, p3 = X[0]
        p12, p13 = X[1]
    else: raise Exception("Invalid type for X.")
        
    # Check bounds on p123 by iterating through all the conditions and checking that 
    # the value of p123 doesn't exceed the bounds.
    mn23,mx23 = 0.,1.
    mn123,mx123 = 0.,1.
    
    # Constraint p123.
    mn123,mx123 = check_bound(mn123,mx123,p12-1,p12)
    mn123,mx123 = check_bound(mn123,mx123,p13-1,p13)
    mn123,mx123 = check_bound(mn123,mx123,-p1+p12+p13,1-p1+p12+p13)
    
    # p23-p123: use p123 to constrain p23
    mn23,mx23 = check_bound(0.+mn123,1.+mx123,
                            mn23-mx123+mn123,mx23-mn123+mx123)
    
    # p2-p23-p12+p123
    check_bound(-mx23+mn123-p2+p12,1-mn23+mx123-p2+p12,0,1);
    # p3-p23-p13+p123
    check_bound(-mx23+mn123-p3+p13,1-mn23+mx123-p3+p13,0,1);
    # 1-p123+p12+p13+p23-p1-p2-p3
    check_bound(-mx123+mn23-1-p12-p13+p1+p2+p3,
                1-mn123+mx23-1-p12-p13+p1+p2+p3,0,1);
    
    return mn23, mx23, mn123, mx123

def check_bound(mn,mx,newmn,newmx):
    """Convenience function for checking bounds."""
    assert mn<=newmx and mx>=newmn, "Bound constraint violated."
    mn = max([mn,newmn])
    mx = min([mx,newmx])
    return mn,mx

def did_all_pairs_vote(X,i,j,k):
    """
    Check if all pairs in pairs of cols i,j,k voted.
    """
    if ( X[:,[i,j]].all(1).any() and X[:,[j,k]].all(1).any() and 
         X[:,[i,k]].all(1).any() ):
        return True
    return False

def check_triplet(X, full_output=False):
    """
    Check one triplet for consistent correlations, i.e. make sure that pij agrees with pik
    and pjk.  All of the random examples below should work since the distribution is well
    defined.

    Parameters
    ----------
    X : ndarray or tuple
        If ndarray, should be of dimensions (n_sample, 3) in terms of {-1,1} and empty
        data points are labeled with 0.  
        If tuple, give <si> then <sisj> in terms of {0,1} basis.
    full_output : bool, False

    Returns
    -------
    bool
        True if distribution is consistent and False if not.
    tuple, optional
    ndarray, optional
    """
    
    if type(X) is np.ndarray:
        assert X.shape[1]==3

        # check all possible triplets
        for ixOrder in [[0,1,2],[1,0,2],[2,0,1]]:
            try:
                pijmn, pijmx, pijkmn, pijkmx = _bounds_case_p23(X[:, ixOrder])
            except AssertionError:
                if full_output:
                    print("Bounds error")
                return False
            pij = (X[:,ixOrder[1:]]==1).all(1).sum() / X[:,ixOrder[1:]].all(1).sum()
            
            # Accounting for numerical precision errors.
            pijmn, pijmx, pij = np.around(pijmn, 10), np.around(pijmx, 10), np.around(pij, 10)
            if not (pijmn<=pij<=pijmx):
                if full_output:
                    return False, (pijmn,pijmx), pij
                return False
        return True

    elif type(X) is tuple:
        assert len(X)==2
        assert (X[0]>=0).all() and (X[1]>=0).all()

        # check all possible orderings of correlations
        for ixOrder1,ixOrder2 in zip([[0,1,2],[1,0,2],[2,0,1]], [[0,1,2],[0,2,1],[1,2,0]]):
            try:
                pijmn, pijmx, pijkmn, pijkmx = _bounds_case_p23(([X[0][i] for i in ixOrder1],
                                                                 [X[1][i] for i in ixOrder2[:2]]))
            except AssertionError:
                if full_output:
                    print("Bounds error")
                return False
            pij = X[1][ixOrder2[-1]]
            
            # Accounting for numerical precision errors.
            pijmn, pijmx, pij = np.around(pijmn, 10), np.around(pijmx, 10), np.around(pij, 10)
            if not (pijmn<=pij<=pijmx):
                if full_output:
                    return False, (pijmn,pijmx), pij
                return False
        return True
    else:
        raise Exception("Invalid type for X.")


# ========================================================= #
# Functions for checking probability constraints generally. #
# ========================================================= #
def A_matrix_for_ising(allstates):
    """
    Determine A matrix for computing the probabilities of all states recursively.
    
    Parameters
    ----------
    allstates : ndarray
        All possible Ising states. These should be ordered in a consistent way probably
        best to get them from entropy.bin_states()
    """
    n = allstates.shape[1]
    assert len(allstates)==2**n
    
    def next_probability_and_sign(coeffs,this_state,add_or_sub):
        """
        Recursive method for filling in the coeffs that belong to each of the marginals
        p_{ij...k}. The marginals are ordered in the same way as the states are output 
        from entropy.bin_states(). In other words, given a state 
        {s1=1,s2=0,s3=1,...,sn=0}, the
        corresponding marginal is p13 = p(s1=s3=1 and other si=0).
        """
        assert len(coeffs)==2**n

        if (this_state==1).all():
            coeffs[-1] += add_or_sub
            return

        stateix = (allstates==this_state[None,:]).all(1)
        coeffs[stateix] += add_or_sub

        # Iterate through all states with one more that is 1.
        zeroix = this_state==0
        nextState = np.zeros(n,dtype=int)

        for substate in bin_states(zeroix.sum()):
            # Only iterate over substates that have at least one more one.
            if any(substate):
                nextState[:] = this_state[:]
                nextState[zeroix] = substate

                next_probability_and_sign(coeffs,nextState,-add_or_sub)
    
    A = np.zeros((2**n,2**n))
    for statei,state in enumerate(allstates):
        coeffs = np.zeros(2**n)
        next_probability_and_sign(coeffs,state,1)
        A[statei,:] = coeffs[:]
        
    A = np.vstack((A,-A))
    return A

def check_joint_consistency(X,full_output=False,linprog_options={},allstates=None):
    """
    Given a binary data set with incomplete data points indicated by 0's, check
    whether or not measurable marginals are compatible with a real probability
    distribution.
    
    Parameters
    ----------
    X : ndarray
        Should be of shape (n_samples, n_spins) where -1 and 1 are up and down and 0 is a hidden spin.
    full_output : bool,False
    linprog_options : dict,{}

    Returns
    -------
    success : bool/dcit
        Returns True if consistent. Can opt to return all details of solution.
    """
    from scipy.optimize import linprog

    n = X.shape[1]
    if allstates is None:
        allstates = bin_states(n)
    pijk = np.zeros(2**n-1)-1  # measured moments, default value -1

    for statei,state in enumerate(allstates[1:]):
        # Check if there are any observations from this subset and calculate pijk if yes.
        subsetIx = state==1
        fullSubsetIx = (X[:,subsetIx]!=0).all(1)

        if np.any(fullSubsetIx):
            if subsetIx.sum()==1:
                pijk[statei] = (X[fullSubsetIx.ravel(),:][:,subsetIx]==1).mean()
            else:
                pijk[statei] = (X[fullSubsetIx,:][:,subsetIx]==1).all(1).mean()
    
    # Make list of bounds for simplex program.
    # First entry corresponding to normalization.
    bounds = [(1,1)]
    for p in pijk:
        # Condition without bounds
        if p==-1:
            bounds.append((0,1))
        # Codition with bounds
        else:
            bounds.append((p,p))

    A = A_matrix_for_ising(allstates)
    soln = linprog(np.zeros(2**n),
                   A_ub=A,
                   b_ub=np.vstack((np.ones(2**n),np.zeros(2**n))),
                   bounds=bounds,
                   options=linprog_options)
    if full_output:
        return soln['success'],soln
    return soln['success']
# End general functions section
