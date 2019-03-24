import numpy as np
import scipy.io as sio
import scipy.linalg as slinalg
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt
import time
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
import sys
import warnings
from DiffusionMaps import *
from LocalPCA import *


def getMahalanobisDists(X, fn_ellipsoid, delta, n_points, rank, maxeigs = None, jacfac = 1.0, verbose=True):
    """
    Compute the Mahalanobis distance between all pairs of points
    Assume that an "equal space" ellipse is a disc of radius "delta"
    in spacetime (since we are trying to find a map back to spacetime).  
    Sample points uniformly at random within that disc, and
    use observations centered at those points to compute Jacobian.
    Parameters
    ----------
    X: ndarray(N, d)
        An array of N points in d dimensions
    fn_ellipsoid: function(int idx, float delta, int n_points) -> ndarray(n_points, d)
        A function which returns a centered ellipsoid in X space which
        is the result of applying the map from a neighborhood of radius
        delta in the preimage.
        To quote from the Singer/Coifman paper:
        "Suppose that we can identify which data points y (j ) belong to the 
                ellipsoid E y (i) ,δ and which reside outside it"
    delta: float
        Spacetime radius from which to sample
    n_points: int
        Number of points to sample in the disc
    rank: int 
        The intrinsic dimension of the manifold (rank of Jacobian)
    maxeigs: int
        The maximum number of eigenvectors to compute.
        By default d, but can go higher for tighter balls around the 
        data driven neighborhoods used in the mask
    jacfac: float
        Only consider distances between points which are in an ellipsoid scaled
        by this factor around each point
    verbose: boolean
        Whether to print progress and other information
    """
    N = X.shape[0]
    d = X.shape[1]
    if not maxeigs:
        maxeigs = rank
    if maxeigs < rank:
        warnings.warn("maxeigs = %i < rank = %i"%(maxeigs, rank))
        maxeigs = rank
    if maxeigs > d:
        warnings.warn("maxeigs = %i > ambient dimension = %i"%(maxeigs, d))
        maxeigs = d

    ## Step 1: Compute Jacobians
    if verbose:
        print("Computing Jacobians, rank = %i, maxiegs = %i..."%(rank, maxeigs))
    tic = time.time()
    ws = np.zeros((N, maxeigs))
    vs = np.zeros((N, d, maxeigs))
    for i in range(N):
        if verbose and i%100 == 0:
            print("%i of %i"%(i, N))
        Y = fn_ellipsoid(i, delta, n_points)
        C = (Y.T).dot(Y)
        w, v = slinalg.eigh(C, eigvals=(C.shape[1]-maxeigs, C.shape[1]-1))
        # Put largest eigenvectors first
        w = w[::-1]
        v = np.fliplr(v)
        ws[i, :] = w
        vs[i, :, :] = v
    if verbose:
        print("Elapsed Time: %.3g"%(time.time()-tic))
    ## Step 1.5 Estimate intrinsic dimension using eigengaps
    if maxeigs > 1:
        ratios = np.median(ws[:, 0:-1]/ws[:, 1::], 0)
        rank_est = np.argmax(np.sign(ratios+d)) + 1
    else:
        rank_est=1


    ## Step 2: Compute squared Mahalanobis distance between
    ## all pairs of points, as well as a mask for which
    ## pairwise distances should actually be included from it
    gamma = np.zeros((N, N))
    mask = np.ones((N, N))
    #Create a matrix whose ith row and jth column contains
    #the dot product of patch i and eigenvector vjk
    D = np.zeros((N, N, maxeigs))
    for k in range(maxeigs):
        D[:, :, k] = X.dot(vs[:, :, k].T)
    if verbose:
        print("Computing Mahalanobis Distances...")
    tic = time.time()
    # Keep a cumulative sum of projected squared magnitudes
    # of the squared distance vector between i and j
    # onto the eigenvectors in j
    projMagsSqr = np.zeros((N, N)) 
    for k in range(maxeigs):
        Dk = D[:, :, k]
        Dk_diag = np.diag(Dk)
        wk = ws[:, k]
        # ||(yi-yj)*vik||^2 projected onto kth component of i's neighborhood
        pij = (Dk_diag[:, None] - Dk.T)**2
        if k < rank:
            # ||(yi-yj)*vjk||^2 projected onto kth component of j's neighborhood
            pji = (Dk - Dk_diag[None, :])**2
            # Mahalanobis distance contribution
            gamma += pij/wk[:, None]
            gamma += pji/wk[None, :]
        # Mask contribution
        projMagsSqr += pij
        mask *=  pij < wk[:, None]*jacfac
    # Finalize Mahalanobis dist by applying a global scale
    gamma *= 0.5*(delta**2)/(rank+2)
    # Finalize mask by checking that remainder of squared distance
    # components are within the remaining components of the ellipsoid
    DSqr = np.sum(X**2, 1)
    DSqr = DSqr[:, None] + DSqr[None, :] - 2*X.dot(X.T)
    if d-maxeigs > 0:
        wk = ws[:, -1]
        mask *= (DSqr - projMagsSqr) < (d-maxeigs)*wk[:, None]*jacfac
    mask *= mask.T
    # Make sure diagonal of mask is 1
    mask = np.maximum(mask, np.eye(mask.shape[0]))
    if verbose:
        print("Elapsed Time: %.3g"%(time.time()-tic))
    return {'gamma':gamma, 'mask':mask, 'rank_est':rank_est, 'DSqr':DSqr, 'vs':vs, 'ws':ws}









"""###################################################
                MUSHROOM EXAMPLE
###################################################"""

def getMushroom(X):
    Y = np.zeros_like(X)
    Y[:, 0] = X[:, 0] + X[:, 1]**3
    Y[:, 1] = X[:, 1] - X[:, 0]**3
    return Y

def getMushroomEllipsoid(X0, idx, delta, n_points):
    x0 = X0[idx, :]
    x0 = x0[None, :]
    Y = delta*np.random.randn(n_points, 2) + x0
    Y = getMushroom(Y) - getMushroom(x0)
    return Y


def testMahalanobisMushroom():
    np.random.seed(0)
    N = 1000
    X = np.random.rand(N, 2)
    Y = getMushroom(X)
    fn_ellipsoid = lambda idx, delta, n_points: getMushroomEllipsoid(X, idx, delta, n_points)
    res = getMahalanobisDists(Y, fn_ellipsoid, 0.001, 100, 2)
    gamma = res["gamma"]
    dMaxSqrCoeff=0.5

    """
    Using https://github.com/jmbr/diffusion-maps
    """
    c = plt.get_cmap('magma_r')
    C1 = c(np.array(np.round(255.0*X[:, 1]/np.max(X[:, 1])), dtype=np.int32))
    C1 = C1[:, 0:3]
    C2 = c(np.array(np.round(255.0*X[:, 0]/np.max(X[:, 1])), dtype=np.int32))
    C2 = C2[:, 0:3]
    
    t = dMaxSqrCoeff*np.max(gamma)*0.001
    tic = time.time()
    YM = getDiffusionMap(gamma, t, distance_matrix=True, neigs=6, thresh=1e-10)
    print("Elapsed Time: %.3g"%(time.time()-tic))

    embedding = Isomap(n_components=2)
    YIso = embedding.fit_transform(Y)

    plt.figure(figsize=(16, 8))
    plt.subplot(241)
    plt.scatter(X[:, 1], X[:, 0], c=C1)
    plt.axis('equal')
    plt.title("Original, Colored by x")

    plt.subplot(245)
    plt.scatter(X[:, 1], X[:, 0], c=C2)
    plt.axis('equal')
    plt.title("Original, Colored by y")

    plt.subplot(242)
    plt.scatter(Y[:, 0], Y[:, 1], c=C1)
    plt.axis('equal')
    plt.title("Mushroom, Colored by x")

    plt.subplot(246)
    plt.scatter(Y[:, 0], Y[:, 1], c=C2)
    plt.axis('equal')
    plt.title("Mushroom, Colored by y")

    plt.subplot(243)
    plt.scatter(YIso[:, 0], YIso[:, 1], c=C1)
    plt.axis('equal')
    plt.title("ISOMAP, Colored by x")

    plt.subplot(247)
    plt.scatter(YIso[:, 0], YIso[:, 1], c=C2)
    plt.axis('equal')
    plt.title("ISOMAP, Colored by y")

    plt.subplot(244)
    plt.scatter(YM[:, 0], YM[:, 1], c=C1)
    plt.axis('equal')
    plt.title("Mahalanobis, Colored by x")

    plt.subplot(248)
    plt.scatter(YM[:, 0], YM[:, 1], c=C2)
    plt.axis('equal')
    plt.title("Mahalanobis, Colored by y")
    plt.savefig("Mushroom.png")






"""###################################################
            PINCHED CIRCLE EXAMPLE
###################################################"""

def getPinchedCircleParam(t):
    x = np.zeros((t.size, 2))
    x[:, 0] = (1.5 + np.cos(2*t))*np.cos(t)
    x[:, 1] = (1.5 + np.cos(2*t))*np.sin(t)
    #x[:, 0] = np.cos(t)
    #x[:, 1] = np.sin(t)
    x += 0.01*np.random.randn(t.size, 2)
    return x

def getPinchedCircleEllipsoid(t0, idx, delta, n_points):
    x0 = np.array([t0[idx]])
    t = x0 + np.linspace(-1, 1)*delta/2.0
    return getPinchedCircleParam(t) - getPinchedCircleParam(x0)

def testMahalanobisCircle():
    dMaxSqrCoeff = 1.0
    np.random.seed(0)
    N = 1000
    t =np.linspace(0, 1, N+1)[0:N]
    t *= 2*np.pi
    Y = getPinchedCircleParam(t)
    fn_ellipsoid = lambda idx, delta, n_points: getPinchedCircleEllipsoid(t, idx, delta, n_points)
    np.random.seed(2)
    res = getMahalanobisDists(Y, fn_ellipsoid, delta=0.1, n_points=100, \
                                rank=1, maxeigs=2, jacfac=10)
    gamma = res["gamma"]
    mask = res["mask"]


    ## Step 1: Show the effect of the mask
    plt.figure(figsize=(8, 8))
    plt.subplot(221)
    plt.imshow(res["DSqr"])
    plt.title("Original")
    plt.subplot(222)
    plt.imshow(res["gamma"])
    plt.title("Full Mahalanobis")
    plt.subplot(223)
    plt.imshow(mask)
    plt.title("Mask")
    plt.subplot(224)
    D = np.array(gamma)
    D[mask == 0] = np.inf
    plt.imshow(D)
    plt.title("Masked Mahalanobis")
    plt.savefig("PinchedCircle_Mask.png", bbox_inches='tight')


    c = plt.get_cmap('magma_r')
    C1 = c(np.array(np.round(255.0*t/np.max(t)), dtype=np.int32))
    C1 = C1[:, 0:3]
    t = dMaxSqrCoeff*np.max(gamma)*0.001
    tic = time.time()
    YMask = getDiffusionMap(gamma, t, mask=mask, distance_matrix=True, neigs=6, thresh=1e-10)
    YNoMask = getDiffusionMap(gamma, t, distance_matrix=True, neigs=6, thresh=1e-10)
    print("Elapsed Time Diffusion Maps: %.3g"%(time.time()-tic))
    
    embedding = Isomap(n_components=2)
    YIso = embedding.fit_transform(Y)

    plt.figure(figsize=(8, 8))

    plt.subplot(221)
    plt.scatter(Y[:, 0], Y[:, 1], c=C1)
    plt.axis('equal')
    plt.title("Warped, Colored by t")

    plt.subplot(222)
    plt.scatter(YIso[:, 0], YIso[:, 1], c=C1)
    plt.axis('equal')
    plt.title("ISOMAP, Colored by t")

    plt.subplot(223)
    plt.scatter(YNoMask[:, 0], YNoMask[:, 1], c=C1)
    plt.axis('equal')
    plt.title("Mahalanobis, Colored by t")

    plt.subplot(224)
    plt.scatter(YMask[:, 0], YMask[:, 1], c=C1)
    plt.axis('equal')
    plt.title("Masked Mahalanobis, Colored by t")

    plt.savefig("PinchedCircle.png", bbox_inches='tight')




"""###################################################
                SPHERE EXAMPLE
###################################################"""

def getSphere(N, dim):
    X = np.random.randn(N, dim)
    XNorm = np.sqrt(np.sum(X**2, 1))
    XNorm[XNorm == 0] = 1
    X /= XNorm[:, None]
    return X

def getSphereEllipsoid(X0, dim, idx, delta, n_points):
    """
    Do rejection sampling, using delta as angle neighborhood
    """
    x0 = X0[idx, :].flatten()
    N = int(np.round(n_points*2*np.pi/delta))
    Z = np.zeros((0, dim))
    while Z.shape[0] < n_points:
        X = getSphere(N, dim)
        dot = X.dot(x0[:, None]).flatten()
        dot[dot < 0] = 0
        dot[dot > 1] = 1
        angles = np.arccos(dot)
        X = X[angles <= delta, :]
        Z = np.concatenate((Z, X), 0)
    return Z[0:n_points, :] - x0[None, :]

def testMahalanobisSphere():
    import cechmate as cm
    from ripser import ripser
    from persim import plot_diagrams as plot_dgms
    dMaxSqrCoeff = 1.0
    N = 500
    dim = 3
    X0 = getSphere(N, dim)

    fn_ellipsoid = lambda idx, delta, n_points: getSphereEllipsoid(X0, dim, idx, delta, n_points)
    np.random.seed(2)
    res = getMahalanobisDists(X0, fn_ellipsoid, delta=0.1, n_points=100, \
                                rank=2, maxeigs=3, jacfac=100)
    gamma = res["gamma"]
    mask = res["mask"]
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(gamma)
    plt.subplot(122)
    plt.imshow(mask)
    plt.show()
    t = dMaxSqrCoeff*np.max(gamma)*0.1
    tic = time.time()
    YNoMask = getDiffusionMap(gamma, t, distance_matrix=True, neigs=6, thresh=1e-10)
    print("Elapsed Time Diffusion Maps no mask: %.3g"%(time.time()-tic))
    tic = time.time()
    YMask = getDiffusionMap(gamma, t, mask=mask, distance_matrix=True, neigs=6, thresh=1e-10)
    print("Elapsed Time Diffusion Maps: %.3g"%(time.time()-tic))

    plt.figure(figsize=(10, 5))
    ax = plt.gcf().add_subplot(121, projection='3d')
    ax.scatter(YNoMask[:, 0], YNoMask[:, 1], YNoMask[:, 2])
    plt.title("No Mask")
    ax = plt.gcf().add_subplot(122, projection='3d')
    ax.scatter(YMask[:, 0], YMask[:, 1], YMask[:, 2])
    plt.title("Mask")
    plt.show()


    tic = time.time()
    dgms_mask_2 = ripser(YMask, coeff=2, maxdim=2)['dgms']
    print("Elapsed Time TDA mask Z/2Z: %.3g"%(time.time()-tic))
    tic = time.time()
    dgms_mask_3 = ripser(YMask, coeff=3, maxdim=2)['dgms']
    print("Elapsed Time TDA mask Z/3Z: %.3g"%(time.time()-tic))
    tic = time.time()

    dgms_nomask_2 = ripser(YNoMask, coeff=2, maxdim=2)['dgms']
    print("Elapsed Time TDA no mask Z/2Z: %.3g"%(time.time()-tic))
    tic = time.time()
    dgms_nomask_3 = ripser(YNoMask, coeff=3, maxdim=2)['dgms']
    print("Elapsed Time TDA no mask Z/3Z: %.3g"%(time.time()-tic))

    plt.figure(figsize=(10, 10))
    plt.subplot(221)
    plot_dgms(dgms_mask_2)
    plt.title("Mask, Z/2")
    plt.subplot(222)
    plot_dgms(dgms_mask_3)
    plt.title("Mask, Z/3")
    plt.subplot(223)
    plot_dgms(dgms_nomask_2)
    plt.title("No Mask, Z/2")
    plt.subplot(224)
    plot_dgms(dgms_nomask_3)
    plt.title("No Mask, Z/3")
    plt.savefig("SphereDGMS.svg", bbox_inches='tight')
    

"""###################################################
              TIME SERIES EXAMPLE
###################################################"""

def getPulseTrain(NSamples, TMin, TMax, AmpMin, AmpMax):
    """
    Make a pulse train, possibly with some error
    Parameters
    ----------
    NSamples: int
        Total number of samples in the pulse train
    TMin: float
        Min period of a randomly varying period
    TMax: float
        Max period of a randomly varying period
    AmpMin: float
        Min amplitude of a randomly varying amplitude
    AmpMax: float
        Max amplitude of a randomly varying amplitude
    Returns
    -------
    x: ndarray(N)
        The pulse train time series
    """
    x = np.zeros(NSamples)
    x[0] = 1
    i = 0
    while i < NSamples:
        i += TMin + int(np.round(np.random.randn()*(TMax-TMin)))
        if i >= NSamples:
            break
        x[i] = AmpMin + (AmpMax-AmpMin)*np.random.randn()
    return x

def convolveGaussAndAddNoise(x, gaussSigma, noiseSigma):
    """
    Smooth out the time series, then add Gaussian noise
    Parameters
    ----------
    x: ndarray(N)
        Time series
    gaussSigma: float
        Width of gaussian
    noiseSigma: float
        Standard deviation of noise
    Returns
    -------
    y: ndarray(N)
        The smoothed and noised time series
    """
    gaussSigma = int(np.round(gaussSigma*3))
    g = np.exp(-(np.arange(-gaussSigma, gaussSigma+1, dtype=np.float64))**2/(2*gaussSigma**2))
    y = np.convolve(x, g, 'same')
    y = y + noiseSigma*np.random.randn(len(x))
    return y

def testMahalanobisTimeSeries():
    from SlidingWindow import getSlidingWindowNoInterp, SlidingWindowAnimator
    from sklearn.decomposition import PCA
    x1 = getPulseTrain(1000, 160, 160, 1, 1)
    x1 = convolveGaussAndAddNoise(x1, 2, 0.01)
    x2 = np.zeros(x1.shape)
    x2[80::] = getPulseTrain(920, 160, 160, 2, 2)
    x2 = convolveGaussAndAddNoise(x2, 8, 0)
    x = x1 + x2
    x = x[200::]

    win = 70
    dim = 1
    Tau = 1
    dT = 1
    X = getSlidingWindowNoInterp(x, win)
    D = np.sum(X**2, 1)[:, None]
    DSqr = D + D.T - 2*X.dot(X.T)
    Y = doDiffusionMaps(DSqr, X[:, 0], dMaxSqrCoeff=100, do_plot=False)

    fig = plt.figure(figsize=(12, 6))
    a = SlidingWindowAnimator("MahalanobisTimeSeries_DiffusionMaps.mp4", fig, x, Y, dim, Tau, dT, hop=5)

if __name__ == '__main__':
    #testMahalanobisCircle()
    #testMahalanobisMushroom()
    #testMahalanobisSphere()
    testMahalanobisTimeSeries()