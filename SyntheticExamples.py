import numpy as np
import scipy.io as sio
import scipy.linalg as slinalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PDE2D import *
from PatchDescriptors import *

class Parabaloid(PDE2D):
    def __init__(self, M, N):
        ys = np.linspace(0, 1, M)
        xs = np.linspace(0, 1, N)
        self.I = xs[None, :]**2 + ys[:, None]**2

class TorusDist(PDE2D):
    """
    A torus parameterized on [0, 1] x [0, 1], with
    an observation function as the distance to some point
    """
    def __init__(self, M, N, x, tile_x = 1, tile_y = 1, alpha_theta = 1.0, alpha_phi = 1.0, lp = 1):
        """
        Parameters
        ----------
        M: int
            Number of samples along y (phi) in principal square
        N: int
            Number of samples along x (theta) in principal square
        x : ndarray (2)
            Position of observation point (theta, phi)
        tile_x: int
            Number of times to repeat along x
        tile_y: int
            Number of times to repeat along y
        alpha_theta : float
            Weight of metric along the x direction
        alpha_phi : float
            Weight of metric along the y direction
        lp: int
            Use the lp norm
        """
        phi = np.linspace(0, 1, M+1)[0:M]
        theta = np.linspace(0, 1, N+1)[0:N]
        theta, phi = np.meshgrid(theta, phi)
        theta = theta.flatten()
        phi = phi.flatten()
        dx = np.abs(x[0]-theta)
        dx = np.minimum(dx, np.abs(x[0]+1-theta))
        dx = np.minimum(dx, np.abs(x[0]-1-theta))
        dy = np.abs(x[1]-phi)
        dy = np.minimum(dy, np.abs(x[1]+1-phi))
        dy = np.minimum(dy, np.abs(x[1]-1-phi))
        dx = alpha_theta*dx 
        dy = alpha_phi*dy 
        dist = (np.abs(dx)**lp + np.abs(dy)**lp)**(1.0/lp)
        self.I = np.tile(np.reshape(dist, (M, N)), (tile_y, tile_x))

def testMahalanobis(pde, pd = (25, 25), nsamples=(30, 30), dMaxSqr = 1, delta=2, kappa=1, rotate=False, do_mahalanobis=True, d=4, periodic=False, cmap='magma_r'):
    f_patch = lambda x: x
    if rotate:
        f_patch = lambda patches: get_derivative_shells(patches, pd, orders=[0, 1], n_shells=50)
    #f_patch = get_pc_histograms
    #f_patch = lambda patches: get_derivative_shells(patches, pd, orders=[0, 1], n_shells=50)
    #f_patch = lambda patches: get_ftm2d_polar(patches, pd)

    pde.makeObservations(pd=pd, nsamples=nsamples, periodic=periodic, buff=delta, rotate=rotate, f_patch=f_patch)
    Xs = pde.Xs
    Ts = pde.Ts
    N = Xs.size
    mask = np.ones((N, N))
    if do_mahalanobis:
        res = pde.getMahalanobisDists(delta=delta, n_points=100, d=d, kappa=kappa)
        sio.savemat("DSqr.mat", res)
        res = sio.loadmat("DSqr.mat")
        DSqr, mask = res["gamma"], res["mask"]
    else:
        D = np.sum(pde.patches**2, 1)[:, None]
        DSqr = D + D.T - 2*pde.patches.dot(pde.patches.T)
    DSqr[DSqr < 0] = 0

    D = np.sqrt(DSqr)
    D[mask == 0] = np.inf
    Y = doDiffusionMaps(DSqr, Xs, dMaxSqr, do_plot=False, mask=mask)
    #Y = Y[:, 0:2]

    fig = plt.figure(figsize=(18, 6))
    ax = fig.add_subplot(131, projection='3d')
    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=Xs, cmap=cmap)
    plt.title("X")
    ax = fig.add_subplot(132, projection='3d')
    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=Ts, cmap=cmap)
    plt.title("T")
    plt.subplot(133)
    plt.imshow(D, aspect='auto', cmap=cmap)
    plt.show()

    pde.makeVideo(Y, D, skip=1, cmap=cmap)

if __name__ == '__main__':
    #pde = Parabaloid(100, 100)
    pde = TorusDist(50, 100, (0.2, 0.2), tile_y=2, lp=2); nsamples=(30, 30); kappa=0.05
    #pde = TorusDist(35, 100, (0.2, 0.2), alpha_phi=0); nsamples=(1, 500); kappa=0.2
    testMahalanobis(pde, pd=(25, 25), nsamples=nsamples, \
                    dMaxSqr=1000, delta=3, d=2, kappa=kappa,\
                    periodic=True, rotate=False, do_mahalanobis=False)
    plt.show()