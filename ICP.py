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

def getProcrustesAlignment(X, Y, idx, weights=np.array([])):
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
    weights: ndarray(M)
        Weights to use per vertex in correspondences
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
    if weights.size == 0:
        weights = np.ones(X.shape[1])
    Cx = getCentroid(X)
    #Pull out the corresponding points in Y by using numpy
    #indexing notation along the columns
    YCorr = Y[:, idx]
    #Get the centroid of the *corresponding points* in Y
    Cy = getCentroid(YCorr)
    #Subtract the centroid from both X and YCorr with broadcasting
    XC = X - Cx
    YCorrC = YCorr - Cy
    YCorrC *= weights[None, :]
    #Compute the singular value decomposition of YCorrC*XC^T.  Remember, it's
    #the point cloud we want to rotate that goes on the right (a common mistake
    #was to flip these)
    (U, S, VT) = np.linalg.svd(YCorrC.dot(XC.T))
    R = U.dot(VT)
    return (Cx, Cy, R)    

def doICP(X, Y, MaxIters, corresp=np.array([]), weights = np.array([]), verbose=False):
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
    corresp: ndarray(L, 2)
        A list of L correspondences
    weights: ndarray(M)
        Weights to use in Procrustes (or empty if uniform weights)
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
    if weights.size == 0:
        weights = np.ones(X.shape[1])
    for i in range(MaxIters):
        #Get the last estimated centroids and rotation matrix using
        #a convenient Python indexing trick
        Cx = CxList[-1]
        Cy = CyList[-1]
        Rx = RxList[-1]
        #Find new correspondences using the current estimate of the alignment
        idx = getCorrespondences(X, Y, Cx, Cy, Rx)
        if corresp.size > 0:
            idx[corresp[:, 0]] = corresp[:, 1]
        #Check for convergence by seeing if all of the correspondences are
        #the same as they were last time
        if np.sum(np.abs(idx-lastidx)) == 0:
            break
        lastidx = idx
        #Perform procrustes alignment using these new correspondences
        (Cx, Cy, Rx) = getProcrustesAlignment(X, Y, idx, weights)
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



def doICP_PDE2D(pde1, Y1, pde2, Y2, corresp = np.array([[]]), weights=np.array([]), initial_guesses=10, MaxIters=100, scale_norm=True, do_plot=False, cmap='magma_r'):
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
    weights: ndarray(M)
        Weights to use in Procrustes (or empty if uniform weights)
    initial_guesses: int
        Number of initial guesses to take
    MaxIters: int
        Maximum number of iterations to perform, regardless of convergence
    scale_norm: boolean
        RMS Normalize point clouds for scale
    do_plot: boolean
        Whether to plot correspondence plot and ICP iterations
    
    Returns
    -------
    {
        'idxMin': list(NIters) of list(M)
            A list of the corresponding indices in pde2 over all iterations
            over the optimal result,
        'rmses_iter': ndarray(NIters)
            RMSEs at each iteration of the optimal result
    }
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
    min_rmse = np.inf
    idxMin = []
    for i in range(initial_guesses):
        S = np.eye(dim)
        if rank < dim and corresp.size == 0:
            # Come up with a random rotation for the subspace
            # that's not determined by the correspondences
            diff = dim-rank
            r, _, _ = np.linalg.svd(np.random.randn(diff, diff))
            S[-diff::, -diff::] = r
        R = U.dot(S.dot(VT))
        Y1i = (Y1 - C1[None, :]).T
        Y2i = (Y2 - C2[None, :]).T
        Y1i = R.dot(Y1i)
        CxList, CyList, RxList, idxList = doICP(Y1i, Y2i, MaxIters, corresp, weights)
        Xs = pde2.Xs[idxList[-1]]
        Ts = pde2.Ts[idxList[-1]]
        rmse = get_rmse(idxList[-1])
        print("rmse=%.3g"%rmse)
        if rmse < min_rmse:
            min_rmse = rmse
            idxMin = idxList

    # Compute RMEs for each step of the best match
    rmses_iter = np.zeros(len(idxMin))
    for i, idx in enumerate(idxMin):
        D2 = getSSM(Y2[idx, :])
        rmses_iter[i] = get_rmse(idx)
    
    ## Step 3: Plot the iterations of the best match

    # Get values of solution image near patch centers
    # for plotting
    f_interp = pde1.getInterpolator()
    patch_centers = f_interp(pde1.Ts.flatten(), pde1.Xs.flatten(), grid=False)

    if do_plot:
        plt.figure(figsize=(12, 8))
        for i, idx in enumerate(idxMin):
            Xs = pde2.Xs[idx]
            Ts = pde2.Ts[idx]
            plt.clf()
            idx = np.array(idx)
            #"""
            plt.subplot(231)
            plt.scatter(pde1.Xs, pde1.Ts, c=Xs, cmap=cmap)
            if corresp.size > 0:
                plt.scatter(pde1.Xs[corresp[:, 0]], pde1.Ts[corresp[:, 0]], 60, 'C2')
            plt.title("Patch Locs Colored By Xs")
            plt.subplot(232)
            plt.scatter(pde1.Xs, pde1.Ts, c=Ts, cmap=cmap)
            if corresp.size > 0:
                plt.scatter(pde1.Xs[corresp[:, 0]], pde1.Ts[corresp[:, 0]], 60, 'C2')
            plt.title("Patch Locs Colored by Ts")
            plt.subplot(233)
            #"""
            #plt.subplot(122)
            plt.plot(rmses_iter)
            plt.scatter([i], [rmses_iter[i]])
            plt.xlabel("Iteration number")
            plt.title("ICP Iteration %i, RMSE=%.3g"%(i, rmses_iter[i]))
            plt.ylabel("RMSE")
            #"""
            plt.subplot(234)
            plt.scatter(Xs, Ts, c=pde1.Xs, cmap=cmap)
            plt.xlabel("Xs Flat Torus")
            plt.ylabel("Ts Flat Torus")
            plt.title("Xs, Ts, Colored By Loc X")
            plt.subplot(235)
            plt.scatter(Xs, Ts, c=pde1.Ts, cmap=cmap)
            plt.xlabel("Xs Flat Torus")
            plt.ylabel("Ts Flat Torus")
            plt.title("Xs, Ts, Colored By Loc T")
            D2 = getSSM(Y2[idx, :])
            plt.subplot(236)
            #"""
            #plt.subplot(121)
            plt.scatter(Xs, Ts, c=patch_centers, cmap='RdGy')
            if corresp.size > 0:
                plt.scatter(Xs[corresp[:, 0]], Ts[corresp[:, 0]], 60, 'C2', marker='x')
            plt.xticks([0, 0.5, 1], ["0", "$\\pi$", "$2 \\pi$"])
            plt.yticks([0, 0.5, 1], ["0", "$\\pi$", "$2 \\pi$"])
            plt.xlabel("$\\theta$")
            plt.ylabel("$\\phi$")
            plt.title("Patch Centers")
            """
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
            """
            plt.savefig("ICP%i.png"%i, bbox_inches='tight')
    
    return {'idxMin':idxMin, 'rmses_iter':rmses_iter}

def ICPConceptFigure():
    np.random.seed(0)
    N = 20
    fac = 0.6
    t1 = np.linspace(0, 1, N)*2*np.pi
    t1 *= fac
    X = np.array([np.cos(t1), np.sin(2*t1)])
    t2 = (np.linspace(0, 1, N)**2)*2*np.pi
    t2 *= fac
    Y = np.array([np.cos(t2), np.sin(2*t2)])
    theta = np.random.rand()
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    Y = R.dot(Y)
    Y += np.array([[0], [3]])
    X, Y = Y, X

    (CxList, CyList, RxList, idxList) = doICP(X, Y, 100)
    NIters = 7
    ncols = int((NIters+1)/2)
    res = 3
    plt.figure(figsize=(res*ncols, res*2))
    plt.subplot(2, ncols, 1)
    plt.scatter(X[0, :], X[1, :])
    plt.scatter(Y[0, :], Y[1, :])
    plt.axis('equal')
    plt.axis('off')
    plt.title("Original Point Clouds")
    for i in range(NIters):
        plt.subplot(2, ncols, 2+i)
        X2 = X - CxList[i]
        Y2 = Y - CyList[i]
        X2 = RxList[i].dot(X2)
        plt.scatter(X2[0, :], X2[1, :])
        plt.scatter(Y2[0, :], Y2[1, :])
        for i1, j1 in enumerate(idxList[i]):
            plt.plot([X2[0, i1], Y2[0, j1]], [X2[1, i1], Y2[1, j1]], 'k')
        plt.axis('equal')
        plt.axis('off')
        plt.title("Iteration %i"%(i+1))
    plt.savefig("ICPConcept.svg", bbox_inches='tight')


if __name__ == '__main__':
    ICPConceptFigure()