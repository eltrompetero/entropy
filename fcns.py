# Functions necessary for calculating entropies.
#
# 2013-06-25

import numpy as np

def estimate_S():
    """
        Estimate the entropy of a partially complete data set (missing voters from some
        votes. Fill in the missing votes with the empirical probabilities of those 
        votes occurring in the data.
        Args:
            votes : array of votes, a vote in each row
    2014-05-22
    """
    for v in votes:
        break
    return

def fill_in_vote(votes):
    """
        For filling in an incomplete data set with missing votes. Take array of votes, 
        and for each vote check whether the vote is complete or not. If it is complete
        then add a count for it. If it isn't complete, go through all possible
        completions of the missing votes given the marginal distributions and give the
        fractional count for each possibility
        NB20140522 
        Args: 
            votes : array of votes, a vote on each row
        Value:
            filledVotes : list of duplets with first element the vote registered and the
                second the fractional count of that vote
    2014-05-22
    """
    filledVotes = []
    for v in votes:
        nanIx = np.argwhere(np.isnan(v)).flatten()
        nanN = nanIx.size
        
        if nanN==0:
            filledVotes.append(( v,1. ))
        else:
            # Just get the complete voting record of the missing people.
            fullVoteIx = np.sum(np.isnan(votes[:,nanIx])==0,1)==nanN
            subVotes = votes[:,nanIx][fullVoteIx,:]
            allStates = get_all_states(nanN)
            
            p = get_state_probs(subVotes,allStates)
            # Now, fill the entries of vote.
            for i in range(2**nanN):
                _vote = v.copy()
                _vote[nanIx] = allStates[i]
                filledVotes.append(( _vote,p[i] ))

    return filledVotes

def SMa(data):
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

def Snaive(samples):
    """
    2014-02-19
    """
    freq = get_state_probs(samples)
    return -np.sum(freq*np.log2(freq))

def calc_p_J(J,n):
    """
        Get probabilities of states given the J's.
    2014-02-08
    """
    from fast import calc_e

    e = calc_e(J,get_allstates(n))
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

def convert_to_maj(states,maj0or1=1):
    """
        Convert given states such that 0 or 1 corresponds to the majority vote. Split
        votes are left as tehy are.
    2014-02-08
    """
    if not np.array_equal( np.unique(states),np.array([0,1]) ):
        raise Exception("This function can only deal with {0,1}.")

    states = states.copy()

    n = states.shape[1]
    ix = np.sum(states,1) < np.ceil(n/2.)
    states[ix,:] = 1-states[ix,:]

    if maj0or1==0:
        states = 1-states

    return states

def get_all_states(n):
    """
        Get all possible binary spin states.
        2013-06-25
    """

    if n<0:
        raise Exception("n cannot be <0")
    if n>15:
        raise Exception("n is too large to enumerate all states.")

    states = np.zeros((2**n,n))
    j=0
    for i in np.ndindex((2,)*n):
        states[j,:] = i
        j+=1
    return states

def squareform(x):
    """
    2014-01-23
    """
    from scipy.spatial import distance
    return distance.squareform(x)

def calc_sisj(data):
    """
    2013-12-19
        Each sample state along a row.
        Value:
            (si,sisj) : duplet of singlet and dubplet means
    """
    (S,N) = data.shape
    sisj = np.zeros(N*(N-1)/2)

    k=0
    for i in range(N-1):
        for j in range(i+1,N):
            sisj[k] = np.mean(data[:,i]*data[:,j])
            k+=1

    return(np.mean(data,0),sisj)

def nan_calc_sisj(data):
    """
    2013-12-19
        Each col in data should correspond to a set votes by one spin.
    """
    (S,N) = data.shape
    sisj = np.zeros(N*(N-1)/2)

    k=0
    for i in range(N-1):
        for j in range(i+1,N):
            sisj[k] = np.nansum(data[:,i]*data[:,j])/ \
                float(np.sum(np.isnan(data[:,i]*data[:,j])==0))
            k+=1

    return ( np.nansum(data,0)/np.sum(np.isnan(data)==0,0), sisj )

def get_state_probs(v,allstates=None):
    """
    Get probability of states given in {0,1} representation.
    2014-02-11
    """
    if not (np.array_equal( np.unique(v),np.array([0,1]) ) or np.all(v==0) or np.all(v==1)):
        raise Exception("Given data array must be in {0,1} representation.")
    from misc_fcns import unique_rows

    n = v.shape[1]
    j=0

    if allstates==None:
        allstates = unique_rows(v,return_inverse=True)
        freq,x = np.histogram( allstates,range(allstates.max()+2) )
    else:
        freq = np.zeros(allstates.shape[0])
        for vote in allstates:
            ix = np.sum( vote==v ,1)==n
            freq[j] = np.sum(ix)
            j+=1
        if np.sum(freq)==0:
            import warnings
            warnings.warn("State not found in given list of all states.")
    freq = freq.astype(float)/np.sum(freq)
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

def calc_sijk(data):
    """
        Calculate sijk in {0,1}.
        2013-06-25
    """
    from itertools import combinations

    n = data.shape[1]
    sijk = np.zeros((n,n,n))

    for i in combinations(range(n),3):
        sijk[i[0],i[1],i[2]] = np.mean(data[:,i[0]]*data[:,i[1]]*data[:,i[2]])

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
