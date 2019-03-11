import numpy as np
import scipy.io as sio
import scipy.linalg as slinalg
import matplotlib.pyplot as plt
import sklearn.feature_extraction.image as skimage
import time
from scipy import interpolate
import scipy.sparse as sparse
from mpl_toolkits.mplot3d import Axes3D
import sys
import warnings
from DiffusionMaps import *
from LocalPCA import *
from PatchDescriptors import *
from PDE2D import *


class PDE3D(object):
    """
    Attributes
    ----------
    I: ndarray(T, X, Y)
        The full 3D PDE spacetime grid
    patches: ndarray(N, d)
        An array of d-dimensional observations
    Xs: ndarray(N)
        X indices of the center of each patch into I
    Ys: ndarray(N)
        Y indices of the center of each patch into I
    Ts: ndarray(N)
        Time indices of the center of each patch into I
    thetas: ndarray(N)
        Angles (in radians) of each patch with respect to 
        an axis-aligned patch
    pd: tuple(int, int, int)
        The dimensions of each patch (time, width, height)
    f: function ndarray->ndarray
        A pointwise homeomorphism to apply to pixels in the observation function
    """
    def __init__(self):
        self.I = np.array([[]])
        self.periodic_x = False
        self.periodic_y = False

    def draw_frame(self, i):
        plt.imshow(self.I[i, :, :], interpolation='none', aspect='auto', cmap='RdGy')
    
    def getInterpolator(self):
        """
        Create a 2D rect bivariate spline interpolator to sample
        from the solution set with interpolation
        """
        ## Step 1: Setup interpolator
        I = self.I
        t = np.arange(I.shape[0])
        x = np.arange(I.shape[1])
        y = np.arange(I.shape[2])
        if self.periodic_x:
            x = x[None, :]*np.ones((3, 1))
            x[0, :] -= I.shape[1]
            x[2, :] += I.shape[1]
            x = x.flatten()
            I = np.concatenate((I, I, I), 1) #Periodic boundary conditions in space
        if self.periodic_y:
            y = y[None, :]*np.ones((3, 1))
            y[0, :] -= I.shape[2]
            y[2, :] += I.shape[2]
            y = y.flatten()
            I = np.concatenate((I, I, I), 2)
        return interpolate.RegularGridInterpolator((t, x, y), I, method="linear")

    def makeObservations(self, pd, nsamples, uniform=True, periodic_x=False, periodic_y=False, rotate = False, \
                            buff = [0.0, 0.0, 0.0], f_pointwise = lambda x: x, f_patch = lambda x: x):
        """
        Make random rectangular observations (possibly with rotation)
        Parameters
        ----------
        pd: tuple(int, int)
            The dimensions of each patch (height, width)
        nsamples: int or tuple(int, int, int)
            The number of patches to sample, or the dimension of the 
            uniform grid from which to sample patches
        uniform: boolean
            Whether to sample the centers uniformly spatially
        periodic: boolean
            Whether to enforce periodic boundary conditions
        rotate: boolean
            Whether to randomly rotate the patches in the XY plane
        buff: tuple(float, float, float)
            A buffer to include around the edges of the patches 
            that are sampled.  If periodic is true, only use
            this buffer in the time axis (the first element)
        f_pointwise: function ndarray->ndarray
            A pointwise homeomorphism to apply to pixels in the observation function
        """
        self.periodic_x = periodic_x
        self.periodic_y = periodic_y
        f_interp = self.getInterpolator()
        # Make sure all patches are within the spacetime limits
        rotstr = ""
        rs = np.array(buff) + np.array(pd)/2.0
        if rotate:
            rotstr = " rotated"
            rs[1::] += np.array(buff[1::]) + np.sqrt((pd[1]/2.0)**2 + (pd[2]/2.0)**2)
            self.rotate_patches = True
        else:
            self.rotate_patches = False
        print("Making %s%s observations of dimension %s..."%(nsamples, rotstr, pd))
        self.pd = pd
        self.f_pointwise = f_pointwise
        self.f_patch = f_patch
        if isinstance(nsamples, tuple):
            T, M, N = nsamples
            if periodic_x:
                x = np.linspace(0, self.I.shape[1], M)
            else:
                x = np.linspace(rs[1], self.I.shape[1]-rs[1], M)
            if periodic_y:
                y = np.linspace(0, self.I.shape[2], N)
            else:
                y = np.linspace(rs[2], self.I.shape[2]-rs[2], N)
            t = np.linspace(rs[0], self.I.shape[0]-rs[0], T)
            t, x, y = np.meshgrid(t, x, y, indexing='ij')
            self.Ts = t.flatten()
            self.Xs = x.flatten()
            self.Ys = y.flatten()
        else:
            # Pick center coordinates of each patch and rotation angle
            if uniform:
                Y = np.random.rand(nsamples*20, 3)
                perm, labmdas = getGreedyPerm(Y, nsamples)
                Y = Y[perm, :]
                self.Ts = rs[0] + Y[:, 0]*(self.I.shape[0]-2*rs[0])
                if periodic_x:
                    self.Xs = Y[:, 1]*self.I.shape[1]
                else:
                    self.Xs = rs[1] + Y[:, 1]*(self.I.shape[1]-2*rs[1])
                if periodic_y:
                    self.Ys = Y[:, 2]*self.I.shape[2]
                else:
                    self.Ys = rs[2] + Y[:, 2]*(self.I.shape[2]-2*rs[2])
            else:
                self.Ts = rs[0]+np.random.rand(N)*(self.I.shape[0]-2*rs[0])
                if periodic_x:
                    self.Xs = np.random.rand(N)*self.I.shape[1]
                else:
                    self.Xs = rs[1] + np.random.rand(N)*(self.I.shape[1]-2*rs[1])
                if periodic_y:
                    self.Ys = np.random.rand(N)*self.I.shape[2]
                else:
                    self.Ys = rs[2] + np.random.rand(N)*(self.I.shape[2]-2*rs[2])
        if rotate:
            self.thetas = np.random.rand(self.Xs.size)*2*np.pi
        else:
            self.thetas = np.zeros_like(self.Xs)

        # Now sample all patches
        pdt, pdx, pdy = np.meshgrid(even_interval(pd[0]), even_interval(pd[1]), even_interval(pd[2]), indexing='ij')
        pdt = pdt.flatten()
        pdx = pdx.flatten()
        pdy = pdy.flatten()

        # Setup all coordinate locations to sample
        cs, ss = np.cos(self.thetas), np.sin(self.thetas)
        xs = cs[:, None]*pdx[None, :] - ss[:, None]*pdy[None, :] + self.Xs[:, None]
        ys = ss[:, None]*pdx[None, :] + cs[:, None]*pdy[None, :] + self.Ys[:, None]
        ts = self.Ts[:, None] + pdt[None, :]
        
        # Use interpolator to sample coordinates for all patches
        coords = np.array([ts.flatten(), xs.flatten(), ys.flatten()]).T
        print(np.min(coords, 0))
        print(np.max(coords, 0))
        patches = f_interp(coords)
        patches = np.reshape(patches, ts.shape)
        self.patches = f_patch(f_pointwise(patches))
        print(self.patches.shape)

    def getMahalanobisDists(self, delta, n_points, d = -10, maxeig = 10, kappa = 1):
        """
        Compute the Mahalanobis distance between all pairs of points
        To quote from the Singer/Coifman paper:
        "Suppose that we can identify which data points y (j ) belong to the 
                ellipsoid E y (i) ,δ and which reside outside it"
        Assume that an "equal space" ellipse is a disc of radius "delta"
        in spacetime (since we are trying to find a map back to spacetime).  
        Sample points uniformly at random within that disc, and
        use observations centered at those points to compute Jacobian.
        Parameters
        ----------
        delta: float
            Spacetime radius from which to sample
        n_points: int
            Number of points to sample in the disc
        d: int (default -10)
            If >0, the dimension of the Jacobian.
            If <0, then pick the dimension at which the eigengap
            exceeds |d|
        maxeig: int (default 10)
            The maximum number of eigenvectors to compute
            (should be at least d)
        kappa: float
            A number in [0, 1] indicating the proportion of mutual nearest
            neighbors to use, in the original patch metric
        """
        pass

    
    def plotPatchBoundary(self, i, draw_arrows=True):
        """
        Plot a rectangular outline of the ith patch
        along with arrows at its center indicating
        the principal axes
        Parameters
        ----------
        i: int
            Index of patch
        """
        pdx = even_interval(self.pd[1])
        pdy = even_interval(self.pd[2])
        x0, x1 = pdx[0], pdx[-1]
        y0, y1 = pdy[0], pdy[-1]
        x = np.array([[x0, y0], [x0, y1], [x1, y1], [x1, y0], [x0, y0]])
        c, s = np.cos(self.thetas[i]), np.sin(self.thetas[i])
        R = np.array([[c, -s], [s, c]])
        xc = self.Xs[i]
        yc = self.Ys[i]
        x = (R.dot(x.T)).T + np.array([[xc, yc]])
        plt.plot(x[:, 0], x[:, 1], 'C0')
        ax = plt.gca()
        R[:, 0] *= self.pd[1]/2
        R[:, 1] *= self.pd[2]/2
        if draw_arrows:
            ax.arrow(xc, yc, R[0, 0], R[1, 0], head_width = 5, head_length = 3, fc = 'c', ec = 'c', width = 1)
            ax.arrow(xc, yc, R[0, 1], R[1, 1], head_width = 5, head_length = 3, fc = 'g', ec = 'g', width = 1)


    def makePatchVideos(self):
        """
        Make a video of the patches
        """
        plt.figure(figsize=(12, 6))
        for i in range(self.patches.shape[0]):
            t = int(np.round(self.Ts[i]))
            patch = np.reshape(self.patches[i, :], self.pd)
            for k in range(patch.shape[0]):
                plt.clf()
                plt.subplot(121)
                self.draw_frame(t)
                self.plotPatchBoundary(i)
                plt.title("Patch %i"%i)
                plt.subplot(122)
                plt.imshow(patch[k, :, :], interpolation='none', aspect='auto', cmap='RdGy')
                plt.title("Frame %i"%k)
                plt.savefig("%i_%i.png"%(i, k))



    def makeVideo(self, Y, D = np.array([]), skip=20, cmap='magma_r'):
        """
        Make a video given a nonlinear dimension reduction, which
        is assumed to be parallel to the patches
        """
        pass