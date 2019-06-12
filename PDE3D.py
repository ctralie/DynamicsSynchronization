import numpy as np
import scipy.io as sio
import scipy.linalg as slinalg
import matplotlib.pyplot as plt
import sklearn.feature_extraction.image as skimage
import time
from scipy import interpolate
import scipy.sparse as sparse
from sklearn.decomposition import PCA
import skimage.transform
from mpl_toolkits.mplot3d import Axes3D
import sys
import warnings
from DiffusionMaps import *
from Mahalanobis import *
from LocalPCA import *
from PatchDescriptors import *
from RotatedPatches import estimate_rotangle
from ConnectionLaplacian import getConnectionLaplacian
import subprocess
from PDEND import *


class PDE3D(PDEND):
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
    f_patch: funtion: ndarray(n_patches, n_pixels) -> ndarray(n_patches, n_features)
        A function to apply to all patches
    pca: sklearn.decomposition.PCA
        A PCA object if the patch function should be composed with
        a linear dimension reduction
    """
    def __init__(self):
        PDEND.__init__(self)
        self.L = np.array([[]])
        self.periodic_x = False
        self.periodic_y = False
        self.mc = None

    def compute_laplacian2D(self):
        """
        Estimate the (complex) slice-wise 2D Laplacian using a 9-stencil, 
        assuming periodic boundary conditions
        """
        if self.L.size > 0:
            # It's already been computed
            return
        self.L = np.zeros_like(self.I)
        weights = {0:-10.0/3, 1:2.0/3, 2:1.0/6}
        for t in range(self.L.shape[0]):
            ft = self.I[t, :, :]
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    index = abs(di) + abs(dj)
                    self.L[t, :, :] += weights[index]*np.roll(np.roll(ft, di, axis=0), dj, axis=1)

    def draw_frame(self, i):
        plt.imshow(self.I[i, :, :], interpolation='none', aspect='auto', cmap='RdGy')
    
    def getInterpolator(self):
        """
        Create a 2D rect bivariate spline interpolator to sample
        from the solution set with interpolation
        """
        if not self.f_interp:
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
            self.f_interp = interpolate.RegularGridInterpolator((t, x, y), I, method="linear")
        return self.f_interp

    def makeObservationsTimeSeries(self, win, hop):
        pass

    def completeObservations(self):
        """
        Sample all of the patches once their positions and orientations
        have been fixed
        """
        tic = time.time()
        f_interp = self.getInterpolator()
        # Now sample all patches
        pd = self.pd
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
        patches = f_interp(coords)
        patches = np.reshape(patches, ts.shape)
        self.patches = self.f_patch(self.f_pointwise(patches))
        print("Elapsed time patch sampling: %.3g"%(time.time()-tic))

    def makeObservations(self, pd, nsamples, uniform=True, periodic_x=False, periodic_y=False, rotate = False, \
                            buff = [0.0, 0.0, 0.0], f_pointwise = lambda x: x, f_patch = lambda x: x):
        """
        Make random rectangular observations (possibly with rotation)
        Parameters
        ----------
        pd: tuple(int, int, int)
            The dimensions of each patch (time, height, width)
        nsamples: int or tuple(int, int, int)
            The number of patches to sample, or the dimension of the 
            uniform grid from which to sample patches
        uniform: boolean
            Whether to sample the centers uniformly spatially
        periodic_x: boolean
            Whether to enforce periodic boundary conditions in x
        periodic_y: boolean
            Whether to enforce periodic boundary conditions in x
        rotate: boolean
            Whether to randomly rotate the patches in the XY plane
        buff: list(float, float, float)
            A buffer to include around the edges of the patches 
            that are sampled.  If periodic is true, only use
            this buffer in the time axis (the first element)
        f_pointwise: function ndarray->ndarray
            A pointwise homeomorphism to apply to pixels in the observation function
        f_patch: function ndarray->ndarray
            A function to apply to each patch to create a new set of observables (e.g. FTM2D)
        """
        self.periodic_x = periodic_x
        self.periodic_y = periodic_y
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
        self.completeObservations()

    def get_mahalanobis_ellipsoid_2DSpatial(self, idx, delta, n_points):
        # function(ndarray(d) x0, int idx, float delta, int n_points) -> ndarray(n_points, d)
        """
        Return a centered ellipsoid at a fixed time, varying the two
        spatial dimensions
        Parameters
        ----------
        idx: int
            Index of patch around which to compute the ellipsoid
        delta: float
            Radius of circle in spacetime to sample in preimage
        n_points: int
            Number of points to sample in the ellipsoid
        Returns
        -------
        ellipsoid: ndarray(n_points, patchdim)
            Centered ellipsoid around patch at index idx
        """
        f_pointwise = self.f_pointwise
        f_patch = self.f_patch
        ## Step 2: For each patch, sample near patches in a disc
        if not self.mc:
            # Cache patch coordinates and interpolator
            pd = self.pd
            pdt, pdx, pdy = np.meshgrid(even_interval(pd[0]), even_interval(pd[1]), even_interval(pd[2]), indexing='ij')
            pdt = pdt.flatten()
            pdx = pdx.flatten()
            pdy = pdy.flatten()
            f_interp = self.getInterpolator()
            self.mc = {'pdx':pdx, 'pdy':pdy, 'pdt':pdt, 'f_interp':f_interp}
        pdx, pdy, pdt, f_interp = self.mc['pdx'], self.mc['pdy'], self.mc['pdt'], self.mc['f_interp']
        y = self.patches[idx, :]
        x0 = self.Xs[idx]
        y0 = self.Ys[idx]
        t0 = self.Ts[idx]
        # Sample centers of each neighboring patch
        # in a disc around the original patch
        xc = x0 + delta*np.random.randn(n_points)
        yc = y0 + delta*np.random.randn(n_points)
        thetasorient = self.thetas[idx]*np.ones(n_points)
        if self.rotate_patches:
            # Randomly rotate each patch if using rotation invariant
            thetasorient = 2*np.pi*np.random.rand(n_points)
        cs = np.cos(-thetasorient)
        ss = np.sin(-thetasorient)
        xs = cs[:, None]*pdx[None, :] - ss[:, None]*pdy[None, :] + xc[:, None]
        ys = ss[:, None]*pdx[None, :] + cs[:, None]*pdy[None, :] + yc[:, None]
        ts = np.ones((n_points, 1)) + pdt[None, :]
        
        # Use interpolator to sample coordinates for all patches
        coords = np.array([ts.flatten(), xs.flatten(), ys.flatten()]).T
        patches = f_interp(coords)
        patches = np.reshape(patches, ts.shape)
        # Apply function and center samples
        Y = f_patch(f_pointwise(patches))
        if self.pca:
            Y = self.pca.transform(Y)
        return Y - y[None, :]

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
        Make a video of each spatiotemporal patch, showing
        where it is on the grid
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

    def plot_complex_3d_hist(self, res, log_density=True):
        """
        Plot an evolving joint histogram of phase and magnitude
        of each frame of the solution, with colors that make it
        easy to see the preimage of the histogram on the domain
        of the image
        Parameters
        ----------
        res: int
            Resolution of histogram
        log_density: boolean
            If true, plot the histogram on a log scale
        """
        if not (self.I.dtype == np.complex):
            print("ERROR: Cannot make complex 3D histogram with dtype: %s"%np.dtype(self.I))
            return
        self.compute_laplacian2D()
        plt.figure(figsize=(12, 12))
        anglebins = np.linspace(-np.pi, np.pi, res+1)
        magbins = np.linspace(np.min(np.abs(self.I)), np.max(np.abs(self.I)), res+1)
        Hs = []
        for i in range(self.I.shape[0]):
            frame = self.I[i, :, :]
            x = frame.flatten()
            xabs = np.abs(x)
            xangle = np.arctan2(np.imag(x), np.real(x))
            H, _, _ = np.histogram2d(xabs, xangle, bins=(magbins, anglebins))
            Hs.append(H)
        Hs = np.array(Hs)
        hmin = np.min(Hs[Hs > 0])
        hmax = np.max(Hs)
        colors2d = scipy.misc.imread("palettes/rotation_2dpalette.png")
        colors2d = scipy.misc.imresize(colors2d, (res, res))
        colors2d = colors2d[:, :, 0:3]
        C = np.reshape(colors2d, (res**2, 3))
        colors2d = colors2d/255.0
        pix = np.arange(res)
        xx, yy = np.meshgrid(anglebins[0:-1], magbins[0:-1])
        for i in range(self.I.shape[0]):
            frame = self.I[i, :, :]
            framel = self.L[i, :, :]
            magframe = np.abs(frame)
            angleframe = np.arctan2(np.imag(framel), np.real(framel))
            frameidx_mag = np.digitize(magframe, magbins)-1
            frameidx_angle = np.digitize(angleframe, anglebins)-1
            frameidx_mag[frameidx_mag>=res] = res-1
            frameidx_angle[frameidx_mag>=res] = res-1
            idx = np.ravel_multi_index((frameidx_mag, frameidx_angle), dims=(res, res))
            cs = C[idx, :]
            cs = np.reshape(cs, (frame.shape[0], frame.shape[1], 3))
            H = Hs[i, :, :]
            plt.clf()
            plt.subplot(221)
            plt.imshow(magframe, vmin=magbins[0], vmax=magbins[-1], cmap='RdGy')
            plt.title("Magnitude")
            plt.subplot(222)
            plt.imshow(angleframe, vmin=-np.pi, vmax=np.pi, cmap='hsv')
            plt.title("Angle")
            ax = plt.gcf().add_subplot(223, projection='3d')
            if log_density:
                H[H == 0] = hmin
                H = np.log(H/hmin)
            ax.plot_surface(xx, yy, H, facecolors=colors2d)
            ax.set_xlabel("Laplacian Angle")
            ax.set_ylabel("W Magnitude")
            if log_density:
                ax.set_zlabel("Log Density")
            else:
                ax.set_zlabel("Density")
            plt.title("Histogram")
            plt.subplot(224)
            plt.imshow(cs)
            plt.title("Activity Mapping")
            plt.savefig("3DHist%i.png"%i)
    
    def plot_complex_distribution(self, res=50):
        if not (self.I.dtype == np.complex):
            print("ERROR: Cannot make complex distribution with dtype: %s"%np.dtype(self.I))
            return
        self.compute_laplacian2D()
        magrange = [np.min(np.abs(self.I)), np.max(np.abs(self.I))]
        reLrange = [np.min(np.real(self.L)), np.max(np.real(self.L))]
        imLrange = [np.min(np.imag(self.L)), np.max(np.imag(self.L))]
        anglebins = np.linspace(-np.pi, np.pi, res+1)
        magbins = np.linspace(np.min(np.abs(self.I)), np.max(np.abs(self.I)), res+1)
        plt.figure(figsize=(18, 6))
        for i in range(self.I.shape[0]):
            frame = self.I[i, :, :]
            x = frame.flatten()
            frameL = self.L[i, :, :]
            xl = frameL.flatten()
            xabs = np.abs(x)
            xlangle = np.arctan2(np.imag(xl), np.real(xl))
            H, _, _ = np.histogram2d(xabs, xlangle, bins=(magbins, anglebins))
            ii, jj = np.unravel_index(np.argmax(H), H.shape)
            realmax = magbins[ii]*np.cos(anglebins[jj])
            imagmax = magbins[ii]*np.sin(anglebins[jj])
            magmax = magbins[ii]


            magframe = np.abs(frame)
            angleframe = np.arctan2(np.imag(frameL), np.real(frameL))
            plt.clf()
            plt.subplot(131)
            plt.imshow(magframe, vmin=magrange[0], vmax=magrange[-1], cmap='RdGy')
            plt.title("Magnitude")
            plt.subplot(132)
            plt.imshow(angleframe, vmin=-np.pi, vmax=np.pi, cmap='hsv')
            plt.title("Laplacian Angle")
            ax = plt.gcf().add_subplot(133, projection='3d')
            ax.scatter(np.real(xl), np.imag(xl), np.abs(x), s=5, zorder=1)
            #ax.scatter([realmax], [imagmax], [magmax], s=100, zorder=2)
            ax.set_xlim(reLrange)
            ax.set_xlabel("Real Lap")
            ax.set_ylim(imLrange)
            ax.set_ylabel("Imag Lap")
            ax.set_zlim(magrange)
            ax.set_zlabel("Magnitude")
            plt.savefig("ComplexDist%i.png"%i, bbox_inches='tight')