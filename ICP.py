"""
Purpose: Code to implement Procrustes Alignment
and the Iterative Closest Points Algorithm
"""
import numpy as np
import scipy
import scipy.spatial
from scipy.spatial import distance


def getCentroid(PC):
    """
    Compute the centroid of a point cloud
    Parameters
    ----------
    PC: ndarray(d, N)
        matrix of points in a point cloud
    Returns
    -------
    C: ndarray(d, 1)
        A matrix of the centroid of the point cloud
    """
    #Take the mean across the rows, make it a column vector
    return np.mean(PC, 1)[:, None] 


def getCorrespondences(X, Y, Cx, Cy, Rx):
    """
    Given an estimate of the aligning matrix Rx that aligns
    X to Y, as well as the centroids of those two point clouds,
    find the nearest neighbors of X to points in Y
    Parameters
    ----------
    X: ndarray(d, M)
        d x M matrix of points in X
    Y: ndarray(d, N)
        d x N matrix of points in Y (the target point cloud)
    Cx: ndarray(d, 1)
        d x 1 matrix of the centroid of X
    Cy: ndarray(d, 1)
        d x 1 matrix of the centroid of corresponding points in Y
    Rx: ndarray(d, d)
        Current estimate of rotation matrix for X
    Returns
    -------
    idx: ndarray(N)
        An array of size N which stores the corresponding indices
    """
    #Center the Y points on the estimated centroid
    YC = Y - Cy
    #Center, then rotate the X points by the estimated rotation to bring X to Y
    XC = X - Cx
    XC = np.dot(Rx, XC)
    #Make a column vector of the dot product of the centered X points with themselves
    dotX = np.sum(XC**2, 0)[:, None]
    #Make a row vector of the dot product of the centered Y points with themselves
    dotY = np.sum(YC**2, 0)[None, :]
    #Use the formula that the squared Euclidean distance between XC^i and YC^j
    #is XC^i dot XC^i + YC^j dot YC^j - 2XC^i dot YC^j
    #And use broadcasting to repeat the column of XC dot XC across the entire
    #matrix and the row of YC dot YC down the entire matrix
    D = dotX + dotY - 2*XC.T.dot(YC)
    idx = np.argmin(D, 1) #Find index of closest point in Y to point in X
    return idx

def getProcrustesAlignment(X, Y, idx):
    """
    Given correspondences between two point clouds, to center
    them on their centroids and compute the Procrustes alignment to
    align one to the other
    Parameters
    ----------
    X: ndarray(d, M)
        d x M matrix of points in X
    Y: ndarray(d, N)
        d x N matrix of points in Y (the target point cloud)
    idx: ndarray(N)
        An array of size M which stores the corresponding indices
    Returns
    -------
    Cx: ndarray(d, 1)
        Matrix of the centroid of X
    Cy: ndarray(d, 1)
        Matrix of the centroid of corresponding points in Y
    Rx: ndarray(d, d)
        A dxd rotation matrix to rotate and align X to Y after
        they have both been centered on their centroids Cx and Cy
    """
    Cx = getCentroid(X)
    #Pull out the corresponding points in Y by using numpy
    #indexing notation along the columns
    YCorr = Y[:, idx]
    #Get the centroid of the *corresponding points* in Y
    Cy = getCentroid(YCorr)
    #Subtract the centroid from both X and YCorr with broadcasting
    XC = X - Cx
    YCorrC = YCorr - Cy
    #Compute the singular value decomposition of YCorrC*XC^T.  Remember, it's
    #the point cloud we want to rotate that goes on the right (a common mistake
    #was to flip these)
    (U, S, VT) = np.linalg.svd(YCorrC.dot(XC.T))
    R = U.dot(VT)
    return (Cx, Cy, R)    

def doICP(X, Y, MaxIters, verbose=False):
    """
    The loop which ties together correspondence finding
    and procrustes alignment to implement the iterative closest points algorithm
    Do until convergence (i.e. as long as the correspondences haven't changed)
    Parameters
    ----------
    X: ndarray(d, M)
        d x M matrix of points in X
    Y: ndarray(d, N)
        d x N matrix of points in Y (the target point cloud)
    MaxIters: int
        Maximum number of iterations to perform, regardless of convergence
    Returns
    -------
    CxList: 
        A list of centroids of X estimated in each iteration (these
        should actually be the same each time)
    CyList: 
        A list of the centroids of corresponding points in Y at each 
        iteration (these might be different as correspondences change)
    RxList: 
        A list of rotations matrices Rx that are produced at each iteration
    idxList: list of lists(M)
        List of evolving correspondences from X to Y
    """
    d = X.shape[0]
    assert(d == Y.shape[0])
    CxList = [np.zeros((d, 1))]
    CyList = [np.zeros((d, 1))]
    RxList = [np.eye(d)]
    idxList = []
    lastidx = np.zeros(X.shape[1])
    for i in range(MaxIters):
        #Get the last estimated centroids and rotation matrix using
        #a convenient Python indexing trick
        Cx = CxList[-1]
        Cy = CyList[-1]
        Rx = RxList[-1]
        #Find new correspondences using the current estimate of the alignment
        idx = getCorrespondences(X, Y, Cx, Cy, Rx)
        #Check for convergence by seeing if all of the correspondences are
        #the same as they were last time
        if np.sum(np.abs(idx-lastidx)) == 0:
            break
        lastidx = idx
        #Perform procrustes alignment using these new correspondences
        (Cx, Cy, Rx) = getProcrustesAlignment(X, Y, idx)
        #Put this new alignment estimate on the back of the lists
        CxList.append(Cx)
        CyList.append(Cy)
        RxList.append(Rx)
        idxList.append(idx)
        if verbose:
            print("Finished iteration %i"%i)
    return (CxList, CyList, RxList, idxList)
