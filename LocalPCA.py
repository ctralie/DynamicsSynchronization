"""
Programmer: Chris Tralie
Purpose: An implementation of local PCA for point clouds in R^d
"""
import numpy as np
import numpy.linalg as linalg
from scipy import sparse 
from scipy.sparse.linalg import lsqr, cg, eigsh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d



def getdists_sqr(X, i, exclude_self=True):
    """
    Get the distances from the ith point in X to the rest of the points
    """
    x = X[i, :]
    x = x[None, :]
    dsqr = np.sum(x**2) + np.sum(X**2, 1)[None, :] - 2*x.dot(X.T)
    dsqr = dsqr.flatten()
    if exclude_self:
        dsqr[i] = np.inf # Exclude the point itself
    return dsqr

def getGreedyPerm(X, N = -1, verbose=True):
    """
    Compute a greedy permutation of the Euclidean points
    Parameters
    ----------
    X: ndarray(N, d)
        Point cloud with N points in d dimensions
    N: int
        Number of points to take in the permutation
    Returns
    -------
    perm: ndarray(N)
        Indices of points in the greedy permutation
    lambdas: ndarray(N)
        Covering radii at different points
    """
    if N == -1:
        N = X.shape[0]
    #By default, takes the first point in the list to be the
    #first point in the permutation, but could be random
    perm = np.zeros(N, dtype=np.int64)
    lambdas = np.zeros(N)
    ds = getdists_sqr(X, 0, exclude_self=False)
    for i in range(1, N):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, getdists_sqr(X, idx, exclude_self=False))
    return (perm, lambdas)

def getLocalPCA(X, eps_pca, gammadim = 0.9, K_usqr = lambda u: np.exp(-5*u)*(u <= 1)):
    """
    Estimate a basis to the tangent plane TxM at every point
    Parameters
    ----------
    X: ndarray(N, p)
        A Euclidean point cloud in p dimensions
    eps_pca: float
        Square of the radius of the neighborhood to consider at each 
        It is assumed that it is such that every point will have at least
        d nearest neighbors, where d is the intrinsic dimension
    gammadim: float
        The explained variance ratio below which to assume all of
        the tangent space is captured.
        Used for estimating the local dimension d
    K_usqr: function float->float
        A C2 positive monotonic decreasing function of a squared argument
        with support on the interval [0, 1]
    
    Returns
    -------
    bases: list of ndarray(p, d)
        All of the orthonormal basis matrices for each point
    """
    N = X.shape[0]
    bases = []
    ds = np.zeros(N) # Local dimension estimates
    for i in range(N):
        dsqr = getdists_sqr(X, i, exclude_self=True)
        Xi = X[dsqr <= eps_pca, :] - X[i, :]
        di = K_usqr(dsqr[dsqr <= eps_pca]/eps_pca)
        if di.size == 0:
            bases.append(np.zeros((Xi.shape[1], 0)))
            continue
        Bi = (Xi*di[:, None]).T # Transpose to be consistent with Singer 
        U, s, _ = linalg.svd(Bi)
        s = s**2
        cumvar_ratio = np.cumsum(s) / np.sum(s)
        ds[i] = np.argmax(cumvar_ratio > gammadim) + 1
        bases.append(U)
    d = int(np.median(ds)) # Dimension estimate
    print(np.mean(ds))
    if (d == 0):
        d = 1
    print("Dimension %i"%d)
    for i, U in enumerate(bases):
        if U.shape[1] < d:
            print("Warning: Insufficient rank for epsilon at point %i"%i)
            # There weren't enough nearest neighbors to determine a basis
            # up to the estimated intrinsic dimension, so recompute with 
            # 2*d nearest neighbors
            dsqr = getdists_sqr(X, i)
            idx = np.argsort(dsqr)[0:min(2*d, X.shape[0])]
            Xi = X[idx, :] - X[i, :]
            U, _, _ = linalg.svd(Xi.T)
            bases[i] = U[:, 0:d]
        else:
            bases[i] = U[:, 0:d]
    return bases