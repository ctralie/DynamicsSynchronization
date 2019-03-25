import numpy as np
import scipy.io as sio
import scipy.linalg as slinalg
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from SlidingWindow import *
from DiffusionMaps import *
from Mahalanobis import *

if __name__ == '__main__':
    x = sio.loadmat("sample_singlecell_tracks_long.mat")["tv"][:, 1]
    x = x[10000:27000]
    x = x[0:8000]
    Tau = 1
    dT = 1
    dim = 200
    X = getSlidingWindowNoInterp(x, dim)

    do_mahalanobis = False
    namestr = "mahalanobis"
    if do_mahalanobis:
        dMaxSqrCoeff = 1.0
        # make mirror symmetric
        y = np.concatenate((x[::-1], x, x[::-1]))
        idx = np.arange(y.size)-x.size
        spl = InterpolatedUnivariateSpline(idx, y)
        fn_ellipsoid = lambda idx, delta, n_points: getTimeSeriesEllipsoid(spl, x.size, X, Tau, dT, idx, delta, n_points)
        res = getMahalanobisDists(X, fn_ellipsoid, delta=10, n_points=100, \
                                    rank=1, maxeigs=11, jacfac=1)
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
        Y = doDiffusionMaps(DSqr, X[:, 0], dMaxSqrCoeff=100, do_plot=False)
    Y = Y[:, 0:2]
    fig = plt.figure(figsize=(12, 6))
    a = SlidingWindowAnimator("MahalanobisTimeSeries_%s.mp4"%namestr, fig, x, Y, dim, Tau, dT, hop=5)