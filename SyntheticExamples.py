import numpy as np
import scipy.io as sio
import scipy.linalg as slinalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PDE2D import *
from PatchDescriptors import *

class Parabaloid(PDE2D):
    def __init__(self, M, N):
        ys = np.linspace(-1, 1, M)
        xs = np.linspace(-1, 1, N)
        self.I = xs[None, :]**2 + ys[:, None]**2

def testMahalanobis():
    #"""
    delta = 2
    p = Parabaloid(100, 100)
    p.makeObservations(pd=(25, 25), nsamples=1000, periodic=False)
    Xs = p.Xs
    Ts = p.Ts
    DSqr = p.getMahalanobisDists(delta=delta, n_points=100, d=2)
    #D = np.sum(p.patches**2, 1)[:, None]
    #DSqr = D + D.T - 2*p.patches.dot(p.patches.T)
    sio.savemat("DSqr.mat", {"DSqr":DSqr, "Xs":Xs, "Ts":Ts})
    #"""

    res = sio.loadmat("DSqr.mat")
    DSqr, Xs, Ts = res['DSqr'], res['Xs'].flatten(), res['Ts'].flatten()
    Y = doDiffusionMaps(DSqr, Xs, 2, do_plot=False)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=Xs, cmap='afmhot')
    plt.title("X")
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=Ts, cmap='afmhot')
    plt.title("T")
    plt.show()

if __name__ == '__main__':
    testMahalanobis()