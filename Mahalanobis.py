import numpy as np
import scipy.io as sio
import scipy.linalg as slinalg
from scipy.interpolate import InterpolatedUnivariateSpline
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
    
    Returns
    -------
    {'gamma': ndarray(N, N)
            The unmaksed squared Mahalanobis distance between all pairs of points
     'mask': ndarray(N, N)
            The binary mask at level maxeigs
     'maskidx': ndarray(N, N)
            The level at which each edge enters in the mask
     'rank_est': int
            Estimated intrinsic dimension of the point cloud 
            (a biproduct of the Mahalanobis algorithm; one can check
            how closely this matches the rank passed in)
     }
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
        if verbose and i%500 == 0:
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
    DSqr = np.sum(X**2, 1)
    DSqr = DSqr[:, None] + DSqr[None, :] - 2*X.dot(X.T)
    gamma = np.zeros((N, N), dtype=np.float32)
    mask = np.ones((N, N), dtype=np.bool)
    # Keep track of the minimum number of eigenvectors at which
    # each edge is removed
    maskidx = maxeigs*np.ones((N, N), dtype=int) 
    #Create a matrix whose ith row and jth column contains
    #the dot product of patch i and eigenvector vjk
    D = np.zeros((N, N, maxeigs), dtype=np.float32)
    for k in range(maxeigs):
        D[:, :, k] = X.dot(vs[:, :, k].T)
    if verbose:
        print("Computing Mahalanobis Distances...")
    tic = time.time()
    # Keep a cumulative sum of projected squared magnitudes
    # of the squared distance vector between i and j
    # onto the eigenvectors in j
    projMagsSqr = np.zeros((N, N), dtype=np.float32) 
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
        # Mask contribution of this level
        projMagsSqr += pij
        mask *=  pij < wk[:, None]*jacfac
        # What the finalized mask would be at this level
        maskf = mask*((DSqr-projMagsSqr) < (d-k-1)*wk[:, None]*jacfac)
        maskidx[maskf == 0] = np.minimum(maskidx[maskf == 0], k)
    
    maskidx = np.minimum(maskidx, maskidx.T)
    # Finalize Mahalanobis dist by applying a global scale
    gamma *= 0.5*(delta**2)/(rank+2)
    #mask = maskidx >= maxeigs
    # Make sure diagonal of mask is 1
    mask = np.maximum(mask, np.eye(mask.shape[0]))
    if verbose:
        print("Elapsed Time: %.3g"%(time.time()-tic))
    return {'gamma':gamma, 'mask':mask, 'maskidx':maskidx, 'rank_est':rank_est}


def getMahalanobisAllThresh(gamma, maskidx, eps, neigs=5, flip=True, maxdim=2, minthresh = 0, maxthresh = np.inf, verbose=False):
    """
    Compute diffusion maps and Rips filtrations at different scales of the mask 
    Parameters
    ----------
    gamma: ndarray(N, N)
        The unmaksed squared Mahalanobis distance between all pairs of points
    maskidx: ndarray(N, N)
        The level at which each edge enters in the mask
    eps: float
        Epsilon for diffusion maps
    neigs: int
        Maximum number of eigenvectors to compute in diffusion maps
    flip: boolean
        By default, the eigenvalues/eigenvectors are sorted in
        increasing order.  If this is true, flip them around,
        and also discard the largest one
    maxdim: int
        Maximum dimension of homology to compute
    minthresh: int
        Minimum threshold to compute
    maxthresh: int
        Maximum threshold to compute
    verbose: boolean
        Whether to print information about the computations
    """
    from ripser import ripser
    Ys = []
    alldgms = []
    tic = time.time()
    for thresh in range(max(0, minthresh), min(np.max(maskidx), maxthresh)):
        if verbose:
            print(thresh, end=' ')
        mask = np.array(maskidx >= thresh, dtype=float)
        Y = getDiffusionMap(gamma, eps=eps, distance_matrix=True, mask=mask, neigs=neigs, flip=flip)
        Ys.append(Y)
        dgms = ripser(Y, n_perm=400, maxdim=2)['dgms']
        alldgms.append(dgms)
    if verbose:
        print("Elapsed Time: %.3g"%(time.time()-tic))
    return {'Ys':Ys, 'alldgms':alldgms}
    

def getTorusPersistenceScores(alldgms, do_plot=False):
    """
    Given a collection persistence diagrams, 
    determine the one which is most likely a torus by looking 
    at H0 to make sure there's only one connected component, 
    and then by looking at the most persistent H2 and the 
    two most persistent H1s.
    Parameters
    ----------
    alldgms: N-length list of list of ndarray(M, 2)
        Persistence diagrams for each point cloud
    Returns
    -------
    {'scores':ndarray(N)
            A list of scores for each point cloud,
    'h2': ndarray(N)
        A list of persistences of the most persistent H2 point,
    'h11': ndarray(N)
        A list of persistences of the most persistent H1 point,
    'h12': ndarray(N)
        A list of persistences of the second most persistent H2 point,
    'h0': ndarray(N)
        A list of the highest non-infinite persistences of H0
    }
    """
    N = len(alldgms)
    h2 = np.zeros(N)
    h11 = np.zeros(N)
    h12 = np.zeros(N)
    h0 = np.zeros(N)
    for i, dgms in enumerate(alldgms):
        h1i = alldgms[i][1]
        h2i = alldgms[i][2]
        if h2i.size > 0:
            h2[i] = np.max(h2i[:, 1]-h2i[:, 0])
        idx = np.argsort(h1i[:, 0]-h1i[:, 1])
        h1i = h1i[idx, :]
        if h1i.size > 0:
            h11[i] = h1i[0, 1] - h1i[0, 0]
        if h1i.shape[0] > 1:
            h12[i] = h1i[1, 1] - h1i[1, 0]
        h0i = dgms[0]
        h0i = h0i[np.isfinite(h0i[:, 1]), :]
        if h0i.size > 0:
            h0[i] = np.max(h0i[:, 1])
    scores = np.array(h12)
    scores[h0 > h12] = 0
    if do_plot:
        plt.plot(h2, 'C2')
        plt.plot(h11, 'C1')
        plt.plot(h12, 'C1', linestyle='--')
        plt.plot(h0, 'C0')
        thresh = 1.2*np.max(h2)
        idx = np.argmax(scores)
        plt.scatter([idx], [h12[idx]], 50, c='C1')
        plt.ylim([0, thresh])
        plt.legend(["$h_2^1$", "$h_1^1$", "$h_1^2$", "$h_0^1$"])
        plt.xlabel("Covariance Threshold $\ell$ for Mask")
        plt.ylabel("Persistence")
        plt.title("Persistences Varying Mahalanobis Mask (Chosen Threshold = %i)"%idx)
    return {'scores':scores, 'h2':h2, 'h11':h11, 'h12':h12, 'h0':h0}
    


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
    Y = np.random.randn(n_points, 2)
    Y /= np.sqrt(np.sum(Y**2, 1))[:, None]
    Y = Y*delta + x0
    Y = getMushroom(Y) - getMushroom(x0)
    return Y


def testMahalanobisMushroom():
    np.random.seed(0)
    N = 5000
    X = np.random.rand(N, 2)
    Y = getMushroom(X)
    fn_ellipsoid = lambda idx, delta, n_points: getMushroomEllipsoid(X, idx, delta, n_points)
    res = getMahalanobisDists(Y, fn_ellipsoid, 0.001, 400, 2)
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
    plt.title("Domain, Colored by x")

    plt.subplot(245)
    plt.scatter(X[:, 1], X[:, 0], c=C2)
    plt.axis('equal')
    plt.title("Domain, Colored by y")

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
    plt.savefig("Mushroom.png", bbox_inches='tight')






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

    plt.figure(figsize=(15, 10))
    ax = plt.gcf().add_subplot(231, projection='3d')
    ax.scatter(YMask[:, 0], YMask[:, 1], YMask[:, 2], c=X0[:, 2])
    plt.title("Mask")
    ax = plt.gcf().add_subplot(234, projection='3d')
    ax.scatter(YNoMask[:, 0], YNoMask[:, 1], YNoMask[:, 2], c=X0[:, 2])
    plt.title("No Mask")

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

    plt.subplot(232)
    plot_dgms(dgms_mask_2)
    plt.title("Mask, $\mathbb{Z}/2$")
    plt.subplot(233)
    plot_dgms(dgms_mask_3)
    plt.title("Mask, $\mathbb{Z}/3$")
    plt.subplot(235)
    plot_dgms(dgms_nomask_2)
    plt.title("No Mask, $\mathbb{Z}/2$")
    plt.subplot(236)
    plot_dgms(dgms_nomask_3)
    plt.title("No Mask, $\mathbb{Z}/3$")

    plt.show()

    #plt.savefig("SphereDGMS.svg", bbox_inches='tight')
    

"""###################################################
              TIME SERIES EXAMPLE
###################################################"""

def getTimeSeriesEllipsoid(spl, N, X, Tau, dT, idx, delta, n_points):
    """
    Compute the ellipsoid of a small neighborhood of a sliding
    window time series
    Parameters
    ----------
    spl: InterpolatedUnivariateSpline
        Spline to interpolate the original time series
    N: int
        Length of original time series
    X: ndarray(M, dim)
        Sliding window time series
    Tau: float
        Lag between samples in each window
    dT: float
        Lag between windows
    idx: int
        Index of window for which to compute ellipsoid
    delta: float
        Time radius of ellipsoid
    n_points: int
        Number of windows to take
    Returns
    -------
    Y: ndarray(n_points, dim)
        Centered ellipsoid in sliding window space, with
        a maximum number of n_points (or fewer if it's 
        near the boundary of the time series)
    """
    dim = X.shape[1]
    ts = dT*idx + np.linspace(-delta, delta, n_points)
    ts = ts[:, None] + (Tau*np.arange(dim))[None, :]
    shape = ts.shape
    ts = ts.flatten()
    Y = np.reshape(spl(ts), shape)
    Y -= X[idx, :]
    return Y

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
    from SlidingWindow import getSlidingWindow, SlidingWindowAnimator
    from sklearn.decomposition import PCA
    np.random.seed(0)

    #"""
    x1 = getPulseTrain(1000, 160, 160, 1, 1)
    x1 = convolveGaussAndAddNoise(x1, 2, 0)
    x2 = np.zeros(x1.shape)
    x2[80::] = getPulseTrain(920, 160, 160, 4, 4)
    x2 = convolveGaussAndAddNoise(x2, 8, 0)
    x = x1 + x2
    x = x[175::]
    x += 0.001*np.random.randn(x.size)
    #"""

    """
    t = np.linspace(0, 8*np.pi, 400)
    x = np.cos(t)
    x += 0.001*np.random.randn(x.size)
    """

    dim = 200
    Tau = 1
    dT = np.pi/3
    X = getSlidingWindow(x, dim, Tau, dT)

    do_mahalanobis = True
    namestr = "mahalanobis"
    if do_mahalanobis:
        dMaxSqrCoeff = 1.0
        # make mirror symmetric
        y = np.concatenate((x[::-1], x, x[::-1]))
        idx = np.arange(y.size)-x.size
        spl = InterpolatedUnivariateSpline(idx, y)
        fn_ellipsoid = lambda idx, delta, n_points: getTimeSeriesEllipsoid(spl, x.size, X, Tau, dT, idx, delta, n_points)
        res = getMahalanobisDists(X, fn_ellipsoid, delta=4, n_points=100, \
                                    rank=1, maxeigs=10, jacfac=1)
        gamma = res["gamma"]
        mask = res["mask"]
        t = dMaxSqrCoeff*np.max(gamma)
        tic = time.time()
        Y = getDiffusionMap(gamma, t, mask=mask, distance_matrix=True, neigs=6, thresh=1e-10)
        print("Elapsed Time Diffusion Maps: %.3g"%(time.time()-tic))

        gammadisp = np.array(gamma)
        gammadisp[mask == 0] = np.nan
        plt.figure(figsize=(12, 12))
        plt.subplot(221)
        plt.plot(x)
        plt.subplot(223)
        plt.imshow(gamma)
        plt.subplot(224)
        plt.imshow(gammadisp)
        ax = plt.gcf().add_subplot(2, 2, 2, projection='3d')
        ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=np.arange(Y.shape[0]), cmap='Spectral')
        plt.show()
    else:
        namestr = "DiffusionMaps"
        D = np.sum(X**2, 1)[:, None]
        DSqr = D + D.T - 2*X.dot(X.T)
        eps = np.max(DSqr)
        Y = getDiffusionMap(DSqr, eps, distance_matrix=True, neigs=3)
    Y = Y[:, 0:2]
    fig = plt.figure(figsize=(12, 6))
    a = SlidingWindowAnimator("MahalanobisTimeSeries_%s.mp4"%namestr, fig, x, Y, dim, Tau, dT, hop=2)

if __name__ == '__main__':
    #testMahalanobisCircle()
    #testMahalanobisMushroom()
    #testMahalanobisSphere()
    testMahalanobisTimeSeries()