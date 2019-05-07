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


class FlatTorusIdeal(PDE2D):
    def __init__(self, M, N, K, neighb_cutoff = 1, do_plot = False):
        """
        Construct a dummy 2D PDE whose intrinsic geometry is perfectly
        sampled from the flat torus
        Parameters
        ----------
        M: int
            Number of samples along y (phi) in principal square
        N: int
            Number of samples along x (theta) in principal square
        K: int
            Dimension of diffusion maps space
        Returns
        -------
        Y: ndarray(M*N, K)
            Dimension reduced version of the points
        """
        tic = time.time()
        self.I = np.array([[]]) # Dummy variable for PDE grid
        phi = np.linspace(0, 1, M+1)[0:M]
        theta = np.linspace(0, 1, N+1)[0:N]
        theta, phi = np.meshgrid(theta, phi)
        theta = theta.flatten()
        phi = phi.flatten()
        self.Xs = theta
        self.Ts = phi
        dx = np.abs(theta[:, None]-theta[None, :])
        dx = np.minimum(dx, np.abs(theta[:, None]+1-theta[None, :]))
        dx = np.minimum(dx, np.abs(theta[:, None]-1-theta[None, :]))
        dy = np.abs(phi[:, None]-phi[None, :])
        dy = np.minimum(dy, np.abs(phi[:, None]+1-phi[None, :]))
        dy = np.minimum(dy, np.abs(phi[:, None]-1-phi[None, :]))
        DSqr = dx**2 + dy**2
        DSqr[DSqr < 0] = 0
        mask = np.ones_like(DSqr)
        mask[np.abs(dx) > neighb_cutoff] = 0
        mask[np.abs(dy) > neighb_cutoff] = 0
        Y = doDiffusionMaps(DSqr, theta, mask=mask, neigs=K+1, do_plot=do_plot)
        if do_plot:
            plt.figure(figsize=(12, 6))
            plt.subplot(121)
            plt.imshow(largeimg(DSqr), cmap='magma_r')
            plt.subplot(122)
            plt.imshow(largeimg(getSSM(Y)), cmap='magma_r')
            plt.show()
        self.Y = Y

if __name__ == '__main__':
    f = FlatTorusIdeal(60, 60, 8, do_plot=True)