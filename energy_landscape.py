# ====================================================================================== #
# Module for exploring the energy landscape of the Ising model.
# Author: Edward Lee, edlee@alumni.princeton.edu
# ====================================================================================== #
import numpy as np
import multiprocess as mp


class EnergyLandscape():
    """Class for calculating features of energy landscape of Ising model in {-1,1} basis."""
    def __init__(self, calc_observables, multipliers):
        """
        Parameters
        ----------
        calc_observables : function
            Function for calculating observables given samples from the distribution.
        multipliers : ndarray
            Langrangian multipliers.
        """
        self.calc_observables=calc_observables
        self.multipliers=multipliers

    def _find_energy_basin(self,x,J):
        """
        Find the local energy minimum.

        Parameters
        ----------
        x : ndarray
            A single state.
        J : ndarray
            Langrangian multipliers.
        """
        x = x.copy()
        xprev = np.zeros(len(x))
        while (x!=xprev).any():
            xprev = x.copy()
            x = self._flip_least_stable_spin(x,J)
        return x
        
    def _flip_least_stable_spin(self,x,J):
        """
        Flip the least stable spin.
        """
        E = -self.calc_observables(x[None,:]).dot(J)
        dE = np.zeros(x.size)
        for i in range(x.size):
            x[i] *= -1
            dE[i] = -self.calc_observables(x[None,:]).dot(J)-E
            x[i] *= -1

        if (dE<0).any():
            x[np.argmin(dE)] *= -1
        return x

    def energy_basins(self,X,multicore=True):
        """Energy basin for each given state.
        
        Parameters
        ----------
        X : ndarray

        Returns
        -------
        energybasin : ndarray
            Each row is the energy basin for the given row of X.
        """
        def f(v):
            return self._find_energy_basin(v,self.multipliers)
        if multicore:
            pool=mp.Pool(mp.cpu_count())
            energybasin=np.vstack(pool.map(f,X))
            pool.close()
        else:
            energybasin=[]
            for v in X:
                energybasin.append(f(v))
            energybasin=np.vstack(energybasin)
        return energybasin
    
    @staticmethod
    def zipf_law(X):
        """Return frequency rank of states.
        
        Parameters
        ----------
        X : ndarray
            (n_samples,n_dim)

        Returns
        -------
        uniqX : ndarray
            Unique states.
        uniqIx : ndarray
            uniqX[uniqIx] recovers given X.
        p : ndarray
            Probability of each unique state.
        """
        from misc.utils import unique_rows

        # Collect unique states.
        uniqIx=unique_rows(X)
        uniqX=X[uniqIx]
        
        p=np.bincount(unique_rows(X,return_inverse=True))
        p=p/p.sum()
        
        # Sort everything by the probability.
        sortIx=np.argsort(p)[::-1]
        p=p[sortIx]
        uniqIx=uniqIx[sortIx]
        uniqX=uniqX[sortIx]
        
        return uniqX,uniqIx,p

