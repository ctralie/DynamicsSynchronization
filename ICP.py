"""
Purpose: Code to implement Procrustes Alignment
and the Iterative Closest Points Algorithm
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.spatial
from scipy.spatial import distance

def getCSM(X, Y):
    """
    Get the cross-similarity matrix between X and Y
    Parameters
    ----------
    X: ndarray(d, M)
        First point cloud
    Y: ndarray(d, N)
        Second point cloud
    """
    XSqr = np.sum(X**2, 0)
    YSqr = np.sum(Y**2, 0)
    DSqr = XSqr[:, None] + YSqr[None, :] - 2*X.T.dot(Y)
    DSqr[DSqr < 0] = 0
    return np.sqrt(DSqr)

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
    D = getCSM(XC, YC)
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

def get_rotation_lowrank(X, Y, scale_norm=True):
    """
    Handle the case where there may or may not be 
    enough correspondences to determine a full rank
    rotation from X to Y
    Parameters
    ----------
    X: ndarray(d, M)
        First point cloud
    Y: ndarray(d, M)
        Second point cloud
    Returns
    -------
    Cx, Cy: ndarray(d), ndarray(d)
        Centroids of point clouds
    U, S, VT: ndarray(d, d), ndarray(d), ndarray(d, d)
        Singular value decomposition
    rank: int
        Estimated rank of SVD
    """
    dim = X.shape[0]
    Cx = getCentroid(X)
    Cy = getCentroid(Y)
    XC = X - Cx
    YC = Y - Cy
    Cov = YC.dot(XC.T)
    (U, S, VT) = np.linalg.svd(Cov)
    eps = np.finfo(Cov.dtype).eps
    thresh = eps*S.max()*max(Cov.shape)
    rank = int(np.sum(S > thresh))
    return Cx.flatten(), Cy.flatten(), U, VT, rank


def get_2d_coverage_image(X, Y, N, xlims=[0, 1], ylims=[0, 1]):
    """
    Return an image which is a rasterized version
    of an indicator function of Xs and Ts
    Parameters
    ----------
    X: ndarray(N)
        X locations
    Y: ndarray(N)
        Y locations
    N: int
        Resolution of image
    """
    I = np.zeros((N, N))
    X -= xlims[0]
    X /= xlims[1]
    X = np.array(np.round(X*N), dtype=int)
    X[X >= N] = N-1
    Y -= ylims[0]
    Y /= ylims[1]
    Y = np.array(np.round(Y*N), dtype=int)
    Y[Y >= N] = N-1
    I[Y, X] = 1
    return I



def doICP_PDE2D(pde1, Y1, pde2, Y2, corresp = np.array([[]]), initial_guesses=10, scale_norm=True, do_plot=False, cmap='magma_r'):
    """
    Do ICP on a set of observations from a PDE
    Parameters
    ----------
    pde1: PDE2D
        First PDE with M patches
    Y1: ndarray(M, k)
        Dimension-reduced version of first set of patches
    pde2: PDE2D
        Second PDE with N patches
    Y2: ndarray(N, k)
        Dimension-reduced version of second set of patches
    corresp: ndarray(L, 2)
        A list of L correspondences
    initial_guesses: int
        Number of initial guesses to take
    scale_norm: boolean
        RMS Normalize point clouds for scale
    do_plot: boolean
        Whether to plot correspondence plot and ICP iterations
    """
    from DiffusionMaps import getSSM
    if scale_norm:
        Y1 -= np.mean(Y1, 0)[None, :]
        Y1 /= np.sqrt(np.mean(np.sum(Y1**2, 1)))
        Y2 -= np.mean(Y2, 0)[None, :]
        Y2 /= np.sqrt(np.mean(np.sum(Y2**2, 1)))
    dim = Y2.shape[1]
    D1 = getSSM(Y1)
    D2 = getSSM(Y2)
    vmax = max(np.max(D1), np.max(D2))
    plt.figure(figsize=(10, 10))
    plt.subplot(221)
    pde1.drawSolutionImage()
    plt.title("Observation 1")
    plt.subplot(222)
    pde2.drawSolutionImage()
    plt.title("Observation 2")
    plt.subplot(223)
    plt.imshow(D1, cmap='magma_r', vmin=0, vmax=vmax)
    plt.colorbar()
    plt.subplot(224)
    plt.imshow(D2, cmap='magma_r', vmin=0, vmax=vmax)
    plt.colorbar()
    plt.savefig("Observations.png", bbox_inches='tight')
    
    D1 = getSSM(Y1)
    get_rmse = lambda idx: np.sqrt(np.mean((D1-getSSM(Y2[idx, :]))**2))
    covdim = int(0.7*np.sqrt(D1.shape[0]))

    ## Step 1: If some correspondences are provided, use them
    # to help come up with a good initial guess
    if corresp.size > 0:
        idx1 = corresp[:, 0]
        idx2 = corresp[:, 1]
        y1 = Y1[idx1, :]
        y2 = Y2[idx2, :]
        C1, C2, U, VT, rank = get_rotation_lowrank(y1.T, y2.T)
        if do_plot:
            x1 = np.concatenate((pde1.Xs[:, None], pde1.Ts[:, None]), 1)
            x2 = np.concatenate((pde2.Xs[:, None], pde2.Ts[:, None]), 1)
            plt.figure(figsize=(12, 6))
            plt.subplot(121)
            plt.scatter(x1[:, 0], x1[:, 1], 20)
            for i in idx1:
                plt.scatter(x1[i, 0], x1[i, 1], 100)
            plt.subplot(122)
            plt.scatter(x2[:, 0], x2[:, 1], 20)
            for i in idx2:
                plt.scatter(x2[i, 0], x2[i, 1], 100)
            plt.savefig("ICP_Correspondences.png", bbox_inches='tight')
    else:
        # Come up with the identity initial rotation which is as good
        # as any other
        C1 = np.zeros(dim)
        C2 = np.zeros(dim)
        U = np.eye(dim)
        VT = np.eye(dim)
        rank = 0

    ## Step 2: Try a bunch of different initial guesses
    if rank == dim:
        # If the rank of the estimated rotation using
        # correspondences is sufficient, then that's as 
        # good a guess as any
        initial_guesses = 1
    max_cov = 0
    idxMax = []
    final_covs = np.zeros(initial_guesses)
    final_rmses = np.zeros(initial_guesses)
    for i in range(initial_guesses):
        S = np.eye(dim)
        if rank < dim:
            # Come up with a random rotation for the subspace
            # that's not determined by the correspondences
            diff = dim-rank
            r, _, _ = np.linalg.svd(np.random.randn(diff, diff))
            S[-diff::, -diff::] = r
        R = U.dot(S.dot(VT))
        Y1i = (Y1 - C1[None, :]).T
        Y2i = (Y2 - C2[None, :]).T
        Y1i = R.dot(Y1i)
        CxList, CyList, RxList, idxList = doICP(Y1i, Y2i, MaxIters=100)
        Xs = pde2.Xs[idxList[-1]]
        Ts = pde2.Ts[idxList[-1]]
        cov_img = get_2d_coverage_image(Xs, Ts, covdim)
        cov = np.sum(cov_img)/float(cov_img.size)
        final_covs[i] = cov
        final_rmses[i] = get_rmse(idxList[-1])
        print("cov=%.3g, rmse=%.3g"%(final_covs[i], final_rmses[i]))
        if cov > max_cov:
            max_cov = cov
            idxMax = idxList
    # Scatterplot relationship between RMSE and coverage
    plt.figure()
    plt.scatter(final_rmses, final_covs)
    plt.xlabel("RMSE")
    plt.ylabel("Coverage")
    plt.title("RMSE vs Coverage")
    plt.savefig("RMSE_Vs_Coverage.png", bbox_inches='tight')

    # Compute RMEs for each step of the best match
    rmses_iter = np.zeros(len(idxMax))
    for i, idx in enumerate(idxMax):
        D2 = getSSM(Y2[idx, :])
        rmses_iter[i] = get_rmse(idx)
    
    ## Step 3: Plot the iterations of the best match
    if do_plot:
        plt.figure(figsize=(15, 15))
        for i, idx in enumerate(idxMax):
            Xs = pde2.Xs[idx]
            Ts = pde2.Ts[idx]
            cov_img = get_2d_coverage_image(Xs, Ts, covdim)
            plt.clf()
            idx = np.array(idx)
            plt.subplot(331)
            plt.scatter(pde1.Xs, pde1.Ts, c=Xs, cmap=cmap)
            plt.title("Patch Locs Colored By Xs")
            plt.subplot(332)
            plt.scatter(pde1.Xs, pde1.Ts, c=Ts, cmap=cmap)
            plt.title("Patch Locs Colored by Ts")
            plt.subplot(333)
            plt.plot(rmses_iter)
            plt.scatter([i], [rmses_iter[i]])
            plt.xlabel("Iteration number")
            plt.title("ICP Convergence")
            plt.ylabel("RMSE")
            plt.subplot(334)
            plt.scatter(Xs, Ts, c=pde1.Xs, cmap=cmap)
            plt.xlabel("Xs Flat Torus")
            plt.ylabel("Ts Flat Torus")
            plt.title("Xs, Ts, Colored By Loc X")
            plt.subplot(335)
            plt.scatter(Xs, Ts, c=pde1.Ts, cmap=cmap)
            plt.xlabel("Xs Flat Torus")
            plt.ylabel("Ts Flat Torus")
            plt.title("Xs, Ts, Colored By Loc T")
            D2 = getSSM(Y2[idx, :])
            plt.subplot(336)
            plt.imshow(cov_img)
            plt.gca().invert_yaxis()
            plt.title("%.3g %% Coverage"%(100*np.sum(cov_img)/float(cov_img.size)))
            plt.subplot(337)
            plt.imshow(D1, cmap=cmap)
            plt.title("D1 Mahalanobis")
            plt.colorbar()
            plt.subplot(338)
            plt.imshow(D2, cmap=cmap)
            plt.colorbar()
            plt.title("D2 Mahalanobis Iter %i"%i)
            plt.subplot(339)
            plt.imshow(D1-D2, cmap=cmap)
            plt.colorbar()
            plt.title("Difference (RMSE=%.3g)"%rmses_iter[i])
            plt.tight_layout()
            plt.savefig("ICP%i.png"%i, bbox_inches='tight')
    
    return {'idxMax':idxMax, 'final_covs':final_covs, 'final_rmses':final_rmses}