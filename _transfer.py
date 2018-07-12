# Module for calculating transfer entropy. Since calculating this requires many parameters, this implemenation
# makes a TransferEntropy class that can hold the parameter and run calculations.

import numpy as np
from sklearn.cluster import KMeans
from multiprocessing import Queue,Array,Process
from misc.fcns import unique_rows

class TransferEntropy(object):
    def __init__(self):
        self.KMEANS_N_JOBS = 2
        self.KMEANS_N_INIT = 100
        self.KMEANS_MAX_ITER = 500
        self.N_WORKERS = 3
        return

    def run_parallel_job( self, workerFunction, workQueue, storage):
        """
        Convenience function for setting up workers and running them.
        2015-12-21
        """
        workers = []
        for i in range(self.N_WORKERS):
            p = Process( target=workerFunction, args=(workQueue,storage,) )
            p.start()
            workers.append( p )

        # Wait for workers to finish then kill workers.
        for w in workers:
            w.join()

    def digitize_vectors( self, m,n_clusters=10 ):
        kmeans = KMeans( n_clusters=n_clusters, 
                         n_jobs=self.KMEANS_N_JOBS, 
                         n_init=self.KMEANS_N_INIT, 
                         max_iter=self.KMEANS_MAX_ITER )
        return kmeans.fit(m).labels_

    def digitize_vector_or_scalar( self, data, numberOfBins ):
        """2015-12-23"""
        if data.shape[1]>1:
            discreteData = self.digitize_vectors( data,numberOfBins )
        else:
            n,bins = np.histogram( data,numberOfBins )
            discreteData = np.digitize( data, bins ).ravel()
        return discreteData
        
    def n_step_transfer_entropy( self, x, y,
                                 kPast=1, kPastOther=1, kFuture=1,
                                 bins=[10,10,10],
                                 returnProbabilities=False):
        """
        Transfer entropy from x->y 
        Using histogram binning for unidimensional data and k-means clustering for k-dimensional data where input data points are a set of points from a trajectory.

        We compute the empirical distribution p(i_{n+1},i_n,j_n) and marginalize over this to get the conditional probabilities required for transfer entropy calculation.

        Note:
        Random seeds with k-means clustering might affect the computed results. Good idea to try several iterations or many different k-means seeds.

        2015-12-23

        Params:
        x
            (n_samples,n_dim)
        y
        kPast (int)
            k steps into the past
        kPastOther (int)
            k steps into the past for other trajectory that we're conditioning on 
        kFuture(int)
            k steps into the future
        [binsPast,binsOtherPast,binsFuture] (list of ints)
            number of bins (or clusters) for trajectories
        returnProbabilities (False, bool)
        """
        kPastMx = max([kPast,kPastOther])

        transferEntropy = 0.

        # Construct matrix of data points (i_{n+1},i_n,j_n) where i and j are vectors.
        future = np.zeros((x.shape[0]-kPastMx-kFuture+1,kFuture))
        past = np.zeros((x.shape[0]-kPastMx-kFuture+1,kPast))
        otherPast = np.zeros((x.shape[0]-kPastMx-kFuture+1,kPastOther))
        for i in range(future.shape[0]):
            future[i,:] = y[(i+kPastMx):(i+kPastMx+kFuture)]
            past[i,:] = y[(i+kPastMx-kPast):(i+kPastMx)]
            otherPast[i,:] = x[(i+kPastMx-kPastOther):(i+kPastMx)]

        discreteFuture = self.digitize_vector_or_scalar( future,bins[2] )
        discretePast = self.digitize_vector_or_scalar( past,bins[0] )
        discreteOtherPast = self.digitize_vector_or_scalar( otherPast,bins[1] )

        xy = np.c_[(discreteFuture,discretePast,discreteOtherPast)]  # data as row vectors arranged into matrix
        uniqxy = xy[unique_rows(xy)]  # unique entries in data that will be assigned probabilities using kernel
        pijk = np.zeros((uniqxy.shape[0]))

        # Compute p(i_{n+1},i_n,j_n)
        for i,row in enumerate(uniqxy):
            pijk[i] = np.sum( np.prod(row[None,:]==xy,1) )
        pijk /= np.sum(pijk)

        # Define functions for multiprocessing. ------------------------------------------------
        def calc_piCondij(i,row,store):
            """p( i_future | i_past, j_past )"""
            ix = np.where( np.prod(row[None,1:]==uniqxy[:,1:],1) )[0]
            p = pijk[ix]
            p /= np.sum(p)

            piCondij = np.sum( p[uniqxy[ix][:,0]==row[0]] )

            # Store result in shared memory access.
            store[i] = piCondij

        def calc_piCondi(i,row,store):
            """p( i_future | i_past )"""
            ix = np.where( row[None,1]==uniqxy[:,1] )[0]
            p = pijk[ix]
            p /= np.sum(p)

            piCondi = np.sum( p[uniqxy[ix][:,0]==row[0]] )

            # Store result in shared memory access.
            store[i] = piCondi

        # Parallelization steps:
        # 1. Create shared memory access for storing results across independent processes.
        # 2. Generate list of tasks in store them in a Queue. Queue must have sentinel (or it holds indefinitely)
        # 3. Generate workers and start them.
        # 4. End workers.

        # Define workers that will take jobs from queue to complete.
        def worker_piCondij(work_queue,storage):
            while True:
                nextTask = work_queue.get()
                if not nextTask is None:
                    nextTask.append(storage)
                    calc_piCondij( *nextTask )
                else:
                    break

        def worker_piCondi(work_queue,storage):
            while True:
                nextTask = work_queue.get()
                if not nextTask is None:
                    nextTask.append(storage)
                    calc_piCondi( *nextTask )
                else:
                    break

        def generate_work_queue():
            # Iterate over all unique data points.
            workQueue = Queue()  # List of jobs to complete.
            for i,row in enumerate(uniqxy):
                workQueue.put( [i,row] )
            # Place one sentinel for each worker so it stops waiting for the queue to fill.
            for i in range(self.N_WORKERS):
                workQueue.put( None )
            return workQueue

        # Memory map for shared memory access to processes.
        # These will store results of computation.
        piCondijStore = Array('d', np.zeros((uniqxy.shape[0])) )
        piCondiStore = Array('d', np.zeros((uniqxy.shape[0])) )

        self.run_parallel_job( worker_piCondij, generate_work_queue(), piCondijStore )
        self.run_parallel_job( worker_piCondi, generate_work_queue(), piCondiStore )

        piCondij = np.array(piCondijStore[:])
        piCondi = np.array(piCondiStore[:])

        transferEntropy = np.nansum( pijk * (np.log2( piCondij ) - np.log2( piCondi )) )
        if returnProbabilities:
            return transferEntropy,pijk,piCondij,piCondi
        return transferEntropy
