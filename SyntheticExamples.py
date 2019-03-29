import numpy as np
import scipy.io as sio
import scipy.linalg as slinalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PDE2D import *
from PatchDescriptors import *
from ICP import *
from Mahalanobis import *

class Parabaloid(PDE2D):
    def __init__(self, M, N):
        PDE2D.__init__(self)
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
        PDE2D.__init__(self)
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

class TorusMultiDist(PDE2D):
    """
    A torus parameterized on [0, 1] x [0, 1], with
    an observation function as the distance to some point
    """
    def __init__(self, M, N, n_points, tile_x = 1, tile_y = 1):
        """
        Parameters
        ----------
        M: int
            Number of samples along y (phi) in principal square
        N: int
            Number of samples along x (theta) in principal square
        n_points: int
            Number of points to take on the torus
        tile_x: int
            Number of times to repeat along x
        tile_y: int
            Number of times to repeat along y
        """
        PDE2D.__init__(self)
        self.M = M
        self.N = N
        self.I = np.array([])
        self.sites = np.random.rand(n_points, 2)
        #self.sites[:, 1] = self.sites[:, 0]
        for i in range(n_points):
            alpha_theta, alpha_phi = np.random.rand(2)
            lp = np.random.randint(1, 3)
            Ti = TorusDist(M, N, self.sites[i, :], tile_x, tile_y, alpha_theta, alpha_phi, lp)
            if i == 0:
                self.I = Ti.I
            else:
                self.I = np.minimum(self.I, Ti.I)
    
    def show_sites(self):
        self.drawSolutionImage()
        plt.scatter(self.sites[:, 0]*self.N, self.sites[:, 1]*self.M)


def testICP(noisefac = 0.001, maxeigs=40, delta=3, jacfac=1, dMaxSqr=10, \
            nsamples1 = (30, 30), nsamples2 = (30, 30), \
            pd1 = (25, 25), pd2 = (25, 25), \
            n_correspondences = 0, initial_guesses = 10, seed=0):
    """
    Align a set of observations to another set of observations using ICP
    """
    np.random.seed(seed)
    pde1 = TorusDist(50, 100, (0.2, 0.2), tile_y=2, lp=2)
    pde1.I += noisefac*np.random.randn(pde1.I.shape[0], pde1.I.shape[1])
    Y1 = testMahalanobis_PDE2D(pde1, pd=pd1, nsamples=nsamples1, \
                    dMaxSqr=dMaxSqr, delta=delta, rank=2, \
                    maxeigs=maxeigs, jacfac=jacfac,\
                    periodic=True, rotate=False, do_mahalanobis=True)
    pde2 = TorusDist(50, 100, (0.2, 0.2), tile_y=2, lp=1)
    pde2.I += noisefac*np.random.randn(pde2.I.shape[0], pde2.I.shape[1])
    Y2 = testMahalanobis_PDE2D(pde2, pd=pd2, nsamples=nsamples2, \
                    dMaxSqr=dMaxSqr, delta=delta, rank=2, \
                    maxeigs=maxeigs, jacfac=jacfac,\
                    periodic=True, rotate=False, do_mahalanobis=True)
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
    get_rmse = lambda idx: np.sqrt(np.mean((D1-getSSM(Y2[idx, :])**2)))

    if n_correspondences > 0:
        # If some correspondences are provided, use them
        # to help come up with a good initial guess
        # Here, just simulate choosing the correspondences
        x1 = np.concatenate((pde1.Xs[:, None], pde1.Ts[:, None]), 1)
        x2 = np.concatenate((pde2.Xs[:, None], pde2.Ts[:, None]), 1)
        C = getCSM(x1.T, x2.T)
        idx1 = np.random.permutation(x1.shape[0])[0:n_correspondences]
        idx2 = np.argmin(C[idx1, :], axis=1)
        y1 = Y1[idx1, :]
        y2 = Y2[idx2, :]
        C1, C2, U, VT, rank = get_rotation_lowrank(y1.T, y2.T)
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.scatter(x1[:, 0], x1[:, 1], 20)
        for i in idx1:
            plt.scatter(x1[i, 0], x1[i, 1], 100)
        plt.subplot(122)
        plt.scatter(x2[:, 0], x2[:, 1], 20)
        for i in idx2:
            plt.scatter(x2[i, 0], x2[i, 1], 100)
        plt.savefig("Correspondences.png", bbox_inches='tight')
    else:
        # Come up with the identity initial rotation which is as good
        # as any other
        C1 = np.zeros(dim)
        C2 = np.zeros(dim)
        U = np.eye(dim)
        VT = np.eye(dim)
        rank = 0

    # Now try a bunch of different initial guesses
    if rank == dim:
        # If the rank of the estimated rotation using
        # correspondences is sufficient, then that's as 
        # good a guess as any
        initial_guesses = 1
    min_rmse = np.inf
    idxsMin = []
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
        rmse = get_rmse(idxList[-1])
        print(rmse)
        if rmse < min_rmse:
            min_rmse = rmse
            idxsMin = idxList
    
    rmses = np.zeros(len(idxList))
    for i, idx in enumerate(idxList):
        D2 = getSSM(Y2[idx, :])
        rmses[i] = get_rmse(idx)
    
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(idxList):
        plt.clf()
        idx = np.array(idx)
        plt.subplot(231)
        plt.scatter(pde1.Xs, pde1.Ts, c=pde2.Xs[idx])
        plt.title("Xs")
        plt.subplot(232)
        plt.scatter(pde1.Xs, pde1.Ts, c=pde2.Ts[idx])
        plt.title("Ts")
        plt.subplot(233)
        plt.plot(rmses)
        plt.scatter([i], [rmses[i]])
        plt.xlabel("Iteration number")
        plt.title("ICP Convergence")
        plt.ylabel("RMSE")

        D2 = getSSM(Y2[idx, :])
        plt.subplot(234)
        plt.imshow(largeimg(D1))
        plt.title("D1 Mahalanobis")
        plt.colorbar()
        plt.subplot(235)
        plt.imshow(largeimg(D2))
        plt.colorbar()
        plt.title("D2 Mahalanobis Iter %i"%i)
        plt.subplot(236)
        plt.imshow(largeimg(D1-D2))
        plt.colorbar()
        plt.title("Difference (RMSE=%.3g)"%rmses[i])
        plt.tight_layout()
        plt.savefig("ICP%i.png"%i, bbox_inches='tight')

def testCylinderMahalanobis():
    pde = TorusDist(35, 100, (0.2, 0.2), alpha_phi=0); nsamples=(1, 500)
    testMahalanobis_PDE2D(pde, pd=(25, 25), nsamples=nsamples, \
                    dMaxSqr=1, delta=3, rank=1, maxeigs=6, jacfac=1,\
                    periodic=True, rotate=False, do_mahalanobis=True, do_plot=True)

def testTorusMahalanobis():
    np.random.seed(6); pde = TorusMultiDist(50, 100, 3, tile_y=2); nsamples=10000 #(30, 30)
    #pde = TorusDist(50, 100, (0.2, 0.2), tile_y=2, lp=1); nsamples=(30, 30)
    noisefac = 0.001
    pde.I += noisefac*np.random.randn(pde.I.shape[0], pde.I.shape[1])
    pde.drawSolutionImage()
    plt.show()
    testMahalanobis_PDE2D(pde, pd=(25, 25), nsamples=nsamples, \
                    dMaxSqr=10, delta=3, rank=2, maxeigs=10, jacfac=2,\
                    periodic=True, rotate=True, do_mahalanobis=True, \
                    precomputed_samples=(400, 400), pca_dim=10, do_plot=True)

if __name__ == '__main__':
    testTorusMahalanobis()
    #testCylinderMahalanobis()
    """
    testICP(nsamples1=1000, nsamples2=2000, noisefac = 0.001, \
            maxeigs=60, delta=3, jacfac=1, dMaxSqr=1, \
            n_correspondences=4, initial_guesses=10, seed=0)
    """