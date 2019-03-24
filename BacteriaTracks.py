import numpy as np
import scipy.io as sio
import scipy.linalg as slinalg
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from SlidingWindow import *
from DiffusionMaps import *

if __name__ == '__main__':
    x = sio.loadmat("sample_singlecell_tracks_long.mat")["tv"][:, 1]
    x = x[10000:27000]
    x = x[0:8000]
    plt.plot(x)
    plt.show()
    win = 200
    X = getSlidingWindowNoInterp(x, win)

    """
    pca = PCA(n_components = 2)
    Y = pca.fit_transform(X)
    eigs = pca.explained_variance_
    """
    #"""
    D = np.sum(X**2, 1)[:, None]
    DSqr = D + D.T - 2*X.dot(X.T)
    Y = doDiffusionMaps(DSqr, X[:, 0], dMaxSqrCoeff=100, do_plot=False)
    #"""
    hop=5
    makeSlidingWindowVideo(Y, hop, win)