import numpy as np
import scipy.io as sio
import scipy.linalg as slinalg
import matplotlib.pyplot as plt
import sklearn.feature_extraction.image as skimage
import time
from scipy import interpolate
import scipy.sparse as sparse
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

class PDE2D(PDEND):
    """
    Attributes
    ----------
    I: ndarray(T, S)
        The full 2D PDE spacetime grid
    patches: ndarray(N, d)
        An array of d-dimensional observations
    Xs: ndarray(N)
        X indices of the center of each patch into I
    Ts: ndarray(N)
        Time indices of the center of each patch into I
    thetas: ndarray(N)
        Angles (in radians) of each patch with respect to 
        an axis-aligned patch
    pd: tuple(int, int)
        The dimensions of each patch (height, width)
    f_pointwise: function ndarray->ndarray
        A pointwise homeomorphism to apply to pixels in the observation function
    f_patch: funtion: ndarray(n_patches, n_pixels) -> ndarray(n_patches, n_features)
        A function to apply to all patches
    pca: sklearn.decomposition.PCA
        A PCA object if the patch function should be composed with
        a linear dimension reduction
    """
    def __init__(self):
        PDEND.__init__(self)
        self.periodic = True
        self.mc = None

    def copy(self):
        """
        Make a copy of this PDE
        """
        other = PDE2D()
        other.I = np.array(self.I)
        other.pd = self.pd
        other.f_pointwise = self.f_pointwise
        other.f_patch = self.f_patch
        other.pca = self.pca
        return other

    def drawSolutionImage(self):
        plt.imshow(self.I, interpolation='none', aspect='auto', cmap='RdGy')

    def getInterpolator(self):
        """
        Create a 2D rect bivariate spline interpolator to sample
        from the solution set with interpolation
        """
        if not self.f_interp:
            ## Step 1: Setup interpolator
            I = self.I
            x = np.arange(I.shape[1])
            t = np.arange(I.shape[0])
            if self.periodic:
                x = x[None, :]*np.ones((3, 1))
                x[0, :] -= I.shape[1]
                x[2, :] += I.shape[1]
                x = x.flatten()
                I = np.concatenate((I, I, I), 1) #Periodic boundary conditions in space
            self.f_interp = interpolate.RectBivariateSpline(t, x, I) # Cache for future use
        return self.f_interp

    def concatenateOther(self, other):
        """
        Add the observations from another system
        This assumes that the same patch dimensions, patch observation
        function, and PCA dimension were used
        Parameters
        ----------
        other: PDE2D
            The other PDE with the same parameters
        """
        self.Xs = np.concatenate((self.Xs, other.Xs))
        self.Ts = np.concatenate((self.Ts, other.Ts))
        self.thetas = np.concatenate((self.thetas, other.thetas))
        self.patches = np.concatenate((self.patches, other.patches), axis=0)


    def completeObservations(self):
        """
        Sample all of the patches once their positions and orientations
        have been fixed
        """
        tic = time.time()
        f_interp = self.getInterpolator()
        pdx, pdt = np.meshgrid(even_interval(self.pd[1]), even_interval(self.pd[0]))
        pdx = pdx.flatten()
        pdt = pdt.flatten()
        ts = np.zeros((self.Xs.size, pdt.size))
        xs = np.zeros((self.Xs.size, pdx.size))

        # Setup all coordinate locations to sample
        cs, ss = np.cos(-self.thetas), np.sin(-self.thetas) # Make CCW wrt "image coordinates"
        xs = cs[:, None]*pdx[None, :] - ss[:, None]*pdt[None, :] + self.Xs[:, None]
        ts = ss[:, None]*pdx[None, :] + cs[:, None]*pdt[None, :] + self.Ts[:, None]
        
        # Use interpolator to sample coordinates for all patches
        patches = (f_interp(ts.flatten(), xs.flatten(), grid=False))
        patches = np.reshape(patches, ts.shape)
        self.patches = self.f_patch(self.f_pointwise(patches))
        print("Elapsed time patch sampling: %.3g"%(time.time()-tic))

    def makeObservations(self, pd, nsamples, uniform=False, periodic=True, rotate = False, \
                            buff = 0.0, f_pointwise = lambda x: x, f_patch = lambda x: x):
        """
        Make random rectangular observations (possibly with rotation)
        Parameters
        ----------
        pd: tuple(int, int)
            The dimensions of each patch (height, width)
        nsamples: int or tuple(int, int)
            The number of patches to sample, or the dimension of the 
            uniform grid from which to sample patches
        uniform: boolean
            Whether to sample the centers uniformly spatially
        periodic: boolean
            Whether to enforce spatial periodic boundary conditions
        rotate: boolean
            Whether to randomly rotate the patches
        buff: float
            A buffer to include around the edges of the patches 
            that are sampled.  If periodic is true, only use
            this buffer in the vertical direction
        f_pointwise: function ndarray->ndarray
            A pointwise homeomorphism to apply to pixels in the observation function
        f_patch: function ndarray->ndarray
            A function to apply to each patch to create a new set of observables (e.g. FTM2D)
        """
        self.periodic = periodic
        # Make sure a rotated patch is within the time range
        # (we usually don't have to worry about space since it's periodic)
        rotstr = ""
        if rotate:
            rotstr = " rotated"
            r = np.sqrt((pd[0]/2)**2 + (pd[1]/2)**2)
            self.rotate_patches = True
        else:
            r = pd[0]/2.0
            self.rotate_patches = False
        r += buff
        print("Making %s%s observations of dimension %s on a grid of %s..."%(nsamples, rotstr, pd, self.I.shape))
        self.pd = pd
        self.f_pointwise = f_pointwise
        self.f_patch = f_patch
        self.pca = None
        if isinstance(nsamples, tuple):
            M, N = nsamples
            if periodic:
                x = np.linspace(0, self.I.shape[1], N)
            else:
                x = np.linspace(r, self.I.shape[1]-r, N)
            t = np.linspace(r, self.I.shape[0]-r, M)
            x, t = np.meshgrid(x, t)
            self.Ts = t.flatten()
            self.Xs = x.flatten()
        else:
            # Pick center coordinates of each patch and rotation angle
            if uniform:
                Y = np.random.rand(nsamples*20, 2)
                perm, labmdas = getGreedyPerm(Y, nsamples)
                Y = Y[perm, :]
                self.Ts = r + Y[:, 0]*(self.I.shape[0]-2*r)
                if periodic:
                    self.Xs = Y[:, 1]*self.I.shape[1]
                else:
                    self.Xs = r + Y[:, 1]*(self.I.shape[1]-2*r)
            else:
                self.Ts = r + np.random.rand(nsamples)*(self.I.shape[0]-2*r)
                if periodic:
                    self.Xs = np.random.rand(nsamples)*self.I.shape[1]
                else:
                    self.Xs = r + np.random.rand(nsamples)*(self.I.shape[1]-2*r)
        if rotate:
            self.thetas = np.random.rand(self.Xs.size)*2*np.pi
        else:
            self.thetas = np.zeros_like(self.Xs)
        
        self.completeObservations()
    
    def get_mahalanobis_ellipsoid(self, idx, delta, n_points):
        # function(ndarray(d) x0, int idx, float delta, int n_points) -> ndarray(n_points, d)
        """
        Return a centered ellipsoid
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
            pdx, pdt = np.meshgrid(even_interval(self.pd[1]), even_interval(self.pd[0]))
            pdx = pdx.flatten()
            pdt = pdt.flatten()
            f_interp = self.getInterpolator()
            self.mc = {'pdx':pdx, 'pdt':pdt, 'f_interp':f_interp}
        pdx, pdt, f_interp = self.mc['pdx'], self.mc['pdt'], self.mc['f_interp']
        y = self.patches[idx, :]
        x0 = self.Xs[idx]
        t0 = self.Ts[idx]
        # Sample centers of each neighboring patch
        # in a disc around the original patch
        tcirc = 2*np.pi*np.linspace(0, 2*np.pi, n_points+1)[0:n_points]
        xc = x0 + delta*np.cos(tcirc)
        tc = t0 + delta*np.sin(tcirc)
        thetasorient = self.thetas[idx]*np.ones(n_points)
        if self.rotate_patches:
            # Randomly rotate each patch if using rotation invariant
            thetasorient = 2*np.pi*np.random.rand(n_points)
        cs = np.cos(-thetasorient)
        ss = np.sin(-thetasorient)
        xs = cs[:, None]*pdx[None, :] - ss[:, None]*pdt[None, :] + xc[:, None]
        ts = ss[:, None]*pdx[None, :] + cs[:, None]*pdt[None, :] + tc[:, None]
        patches = f_interp(ts.flatten(), xs.flatten(), grid=False)
        patches = np.reshape(patches, (n_points, pdx.size))
        # Apply function and center samples
        Y = f_patch(f_pointwise(patches))
        if self.pca:
            Y = self.pca.transform(Y)
        return Y - y[None, :]

    def get_mahalanobis_ellipsoid_from_precomputed(self, other, idx, delta, n_points, delta_theta, verbose=False):
        # function(ndarray(d) x0, int idx, float delta, int n_points) -> ndarray(n_points, d)
        """
        Return a centered ellipsoid, using patches that have already been
        constructed from a system that's assumed to be using the same
        observation functions
        Parameters
        ----------
        other: PDE2D
            Another instance of this PDE, assumed to have been sampled
            on a grid with the same dimensions, using the same patch functions
        idx: int
            Index of patch around which to compute the ellipsoid
        delta: float
            Radius of circle in spacetime to sample in preimage
        n_points: int
            Number of points to sample in the ellipsoid
        delta_theta: float
            The maximum angle allowed between patches that are considered
            neighbors
        Returns
        -------
        ellipsoid: ndarray(n_points, patchdim)
            Centered ellipsoid around patch at index idx
        """
        y = self.patches[idx, :]
        x = np.array([self.Ts[idx], self.Xs[idx]])
        X = np.array([other.Ts, other.Xs]).T
        C = np.sum(x**2) + np.sum(X**2, 1) - 2*np.sum(X*x[None, :], 1)
        if other.periodic:
            X[:, 1] = (X[:, 1] + other.I.shape[1]) % self.I.shape[1]
            C2 = np.sum(x**2) + np.sum(X**2, 1) - 2*np.sum(X*x[None, :], 1)
            C = np.minimum(C, C2)
        Y = other.patches[(C <= delta**2)*(np.abs(other.thetas-self.thetas[idx]) < delta_theta), :]
        if verbose:
            print("Patch %i: %i existing points in jacobian"%(idx, Y.shape[0]))
        if Y.shape[0] > n_points:
            Y = Y[np.random.permutation(Y.shape[0])[0:n_points], :]
        return Y - y[None, :]

    def estimate_rotations(self, X, K, corrweight=False):
        """
        Given estimated diffusion map coordinates, estimate
        rotations by doing pairwise estimates of near neighbors
        and diffusion error out globally using the connection Laplacian
        Parameters
        ----------
        X: ndarray(N, d)
            Estimated diffusion map coordinates
        K: int
            Number of near neighbors to take for each patch
        
        Returns
        -------
        {'thetas_est': ndarray(N)
            Estimated global rotations for each patch,
         'thetasij': ndarray(M, 3)
            Relative estimations between some subset of all pairs}
        """
        self.sample_raw_patches()
        D = getSSM(X)
        N = D.shape[0]
        np.fill_diagonal(D, np.inf)
        J = np.argpartition(D, K, 1)[:, 0:K] # Nearest neighbor indices
        I = np.arange(J.shape[0])[:, None]*np.ones((1, J.shape[1]))
        J = J.flatten()
        I = np.array(I.flatten(), dtype=int)
        B = np.zeros_like(D)
        B[I, J] = 1
        B[J, I] = 1
        ws = []
        Os = []
        thetasij = []
        for i in range(N):
            if i%100 == 0:
                print("%i"%i, end=' ')
            pi = self.get_patch(i)
            fft1 = np.array([])
            for j in range(i+1, N):
                if B[i, j] == 0:
                    continue
                pj = self.get_patch(j)
                res = estimate_rotangle(pi, pj, fft1=fft1)
                fft1 = res['fft1'] # Cache the fft for the first patch
                theta = res['theta_est'] # CCW theta taking pj to pi
                # Oij moves vectors from j to i (so it is opposite direction)
                theta *= -1
                c = np.cos(theta)
                s = np.sin(theta)
                Oij = np.array([[c, -s], [s, c]])
                Os.append(Oij)
                weight = 1.0 
                if corrweight:
                    weight = np.max(res['corr'])
                ws.append([i, j, weight])
                Os.append(Oij.T)
                ws.append([j, i, weight])
                thetasij.append([i, j, theta])
        ws = np.array(ws)
        if corrweight:
            # Normalize weights to the range [0.5, 1]
            ws[:, 2] -= np.min(ws[:, 2])
            ws[:, 2] /= np.max(ws[:, 2])
            ws[:, 2] = ws[:, 2]*0.5 + 0.5
        thetasij = np.array(thetasij)
        print("Doing connection Laplacian...")
        w, v = getConnectionLaplacian(ws, Os, N, 2)
        print(w)
        v = np.reshape(v[:, 0], (N, 2))
        thetas_est = np.arctan2(v[:, 1], v[:, 0])
        self.recover_original_patches()
        return {'thetas_est':thetas_est, 'thetasij':thetasij, 'ws':ws}

    def plotEstimatedRotations(self, thetas, ntoplot=None):
        """
        Plot a vector field showing the difference between
        the true angle and estimated angle
        Parameters
        ----------
        thetas: ndarray(N)
            Estimated thetas (same number as self.thetas)
        ntoplot: int
            An int <= N of numbers of arrows to plot, sampled
            from a greedy permutation based on centers
        """
        N = self.thetas.size
        if not ntoplot:
            idxs = np.arange(N)
        else:
            x = np.array([self.Xs, self.Ts]).T
            idxs, _ = getGreedyPerm(x, N=ntoplot)
        ax = plt.gca()
        for idx in idxs:
            dtheta = self.thetas[idx] - thetas[idx]
            vidx = 4*np.array([np.cos(dtheta), np.sin(dtheta)])
            ax.arrow(self.Xs[idx], self.Ts[idx], vidx[0], vidx[1], head_width=2, head_length=2)
    
    def plotRelativeRotationErrors(self, thetasij):
        """
        Make a scatterplot of the average relative angular error
        at each pixel, with opacity proportional to the error
        Parameters
        ----------
        thetasij: ndarray(M, 3)
            Relative estimations between some subset of all pairs}
        """
        N = self.thetas.size
        errs = np.zeros(N)
        counts = np.zeros(N)
        I = np.array(thetasij[:, 0], dtype=int)
        J = np.array(thetasij[:, 1], dtype=int)
        thetaijest = np.abs(thetasij[:, 2])
        thetaijest = np.minimum(thetaijest, 2*np.pi-thetaijest)
        thetaijgt = np.abs(self.thetas[I] - self.thetas[J])
        thetaijgt = np.minimum(thetaijgt, 2*np.pi-thetaijgt)

        for i in range(thetasij.shape[0]):
            diff = np.abs(thetaijest[i]-thetaijgt[i])
            errs[I[i]] += diff
            errs[J[i]] += diff
            counts[I[i]] += 1
            counts[J[i]] += 1
        errs /= counts
        q5 = np.quantile(errs, 0.05)
        q100 = np.max(errs)
        qs = q5 + (q100-q5)*(np.linspace(0, 1, 10))
        plt.scatter(self.Xs*0, self.Ts*0, 50, c=errs/q100)
        cbar = plt.colorbar(ticks=[q/q100 for q in qs])
        cbar.ax.set_yticklabels(["%.2g"%(180*q/np.pi) for q in qs])
        
        cmap = plt.get_cmap('viridis')
        colors = cmap(errs/q100)
        alpha = np.log(errs/q5)
        alpha[alpha < 0] = 0
        alpha /= np.max(alpha)
        colors[..., -1] = alpha      
        plt.scatter(self.Xs, self.Ts, 60, edgecolors='none', c=colors, marker='.')



    def plotPatchBoundary(self, i, color = 'C0', sz = 1, draw_arrows=True, flip_y = True):
        """
        Plot a rectangular outline of the ith patch
        along with arrows at its center indicating
        the principal axes
        Parameters
        ----------
        i: int
            Index of patch
        draw_arrows: boolean
            Whether to draw a coordinate system in the patch
        color: string
            Color of boundary box
        sz: int
            Thickness of boundary

        flip_y: boolean
            Whether to flip the y arrow for the coordinate system
        """
        pdx = even_interval(self.pd[1])
        pdt = even_interval(self.pd[0])
        x0, x1 = pdx[0], pdx[-1]
        t0, t1 = pdt[0], pdt[-1]
        x = np.array([[x0, t0], [x0, t1], [x1, t1], [x1, t0], [x0, t0]])
        c, s = np.cos(-self.thetas[i]), np.sin(-self.thetas[i])
        R = np.array([[c, -s], [s, c]])
        xc = self.Xs[i]
        tc = self.Ts[i]
        x = (R.dot(x.T)).T + np.array([[xc, tc]])
        plt.plot(x[:, 0], x[:, 1], color, lineWidth=sz)
        ax = plt.gca()
        R[:, 0] *= self.pd[1]/2
        R[:, 1] *= self.pd[0]/2
        if flip_y:
            R[:, 1] *= -1
        if draw_arrows:
            ax.arrow(xc, tc, R[0, 0], R[1, 0], head_width = 5, head_length = 3, fc = 'c', ec = 'c', width = 1)
            ax.arrow(xc, tc, R[0, 1], R[1, 1], head_width = 5, head_length = 3, fc = 'g', ec = 'g', width = 1)
    
    def resort_byidx(self, idx):
        self.Xs = self.Xs[idx]
        self.Ts = self.Ts[idx]
        self.thetas = self.thetas[idx]
        self.patches = self.patches[idx, :]
    
    def resort_byrandom(self):
        idx = np.random.permutation(self.Xs.size)
        self.resort_byidx(idx)
        return idx

    def resort_byraster(self, resy=20):
        idx = approximate_rasterorder(self.Xs, self.Ts, resy)
        self.resort_byidx(idx)
        return idx

    def get_patch(self, idx):
        """
        Unwrap a patch before a function was applied
        """
        p = self.patches[idx, :]
        return np.reshape(p, self.pd)

    def plotPatches(self, n_patches = None, save_frames = True):
        """
        Make a video of the patches
        """
        print(np.min(self.Ts))
        if save_frames:
            plt.figure(figsize=(12, 6))
        else:
            self.drawSolutionImage()
        for i in range(self.patches.shape[0]):
            if save_frames:
                plt.clf()
                plt.subplot(121)
                self.drawSolutionImage()
            self.plotPatchBoundary(i)
            if save_frames or i == self.patches.shape[0]-1:
                plt.axis('equal')
                plt.xlim(0, self.I.shape[1])
                plt.ylim(self.I.shape[0], 0)
            if save_frames:
                plt.subplot(122)
                p = self.get_patch(i)
                plt.imshow(p, interpolation='none', cmap='RdGy', vmin=np.min(self.I), vmax=np.max(self.I))
                plt.title("%.3g degrees"%(self.thetas[i]*180/np.pi))
                plt.savefig("%i.png"%i, bbox_inches='tight')

    def makeVideo(self, Y, D = np.array([]), skip=20, cmap='viridis', colorvar1=np.array([]), colorvar2 = np.array([])):
        """
        Make a video given a nonlinear dimension reduction, which
        is assumed to be indexed parallel to the patches and in approximate raster order
        """
        # Resample original patches for display
        self.sample_raw_patches()
        if colorvar1.size == 0:
            colorvar1 = self.Xs/np.max(self.Xs)
        if colorvar2.size == 0:
            colorvar2 = self.Ts/np.max(self.Ts)
        c = plt.get_cmap(cmap)
        C1 = c(np.array(np.round(255.0*colorvar1), dtype=np.int32))
        C1 = C1[:, 0:3]
        C2 = c(np.array(np.round(255.0*colorvar2), dtype=np.int32))
        C2 = C2[:, 0:3]        
        res = 6
        ncols = 2
        if D.size > 0:
            ncols = 3
        fig = plt.figure(figsize=(res*ncols, res*2))
        I, Xs, Ts = self.I, self.Xs, self.Ts
        idx = 0
        N = Y.shape[0]
        for i in range(0, N, skip):
            plt.clf()
            plt.subplot(2, ncols, 1)
            self.drawSolutionImage()
            # Preserve x and y lims
            xlims = plt.gca().get_xlim()
            ylims = plt.gca().get_ylim()
            self.plotPatchBoundary(i)
            plt.xlabel("Space")
            plt.ylabel("Time")
            plt.axis('equal')
            plt.xlim(xlims)
            plt.ylim(ylims)
            plt.subplot(2, ncols, 2)
            p = self.get_patch(i)
            plt.imshow(p, interpolation='none', cmap='RdGy', vmin=np.min(self.I), vmax=np.max(self.I))
            plt.axis('off')
            if Y.shape[1] == 2:
                plt.subplot(2, ncols, ncols+1)
                plt.scatter(Y[:, 0], Y[:, 1], 100, c=np.array([[0, 0, 0, 0]]))
                plt.scatter(Y[0:i+1, 0], Y[0:i+1, 1], 20, c=C1[0:i+1, :])
                plt.scatter(Y[i, 0], Y[i, 1], 40, 'r')
                plt.axis('equal')
                ax = plt.gca()
                ax.set_facecolor((0.15, 0.15, 0.15))
                ax.set_xticks([])
                ax.set_yticks([])
            elif Y.shape[1] == 3:
                ax = plt.gcf().add_subplot(200+ncols*10+ncols+1, projection='3d')
                ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=np.array([[0, 0, 0, 0]]))
                ax.scatter(Y[0:i+1, 0], Y[0:i+1, 1], Y[0:i+1, 2], c=C1[0:i+1, :])
                ax.scatter(Y[i, 0], Y[i, 1], Y[i, 2], 'r')
            else:
                plt.subplot(2, ncols, ncols+1)
                plt.scatter(Y[:, 0], Y[:, 1], 100, c=np.array([[0, 0, 0, 0]]))
                plt.scatter(Y[0:i+1, 0], Y[0:i+1, 1], 20, c=C1[0:i+1, :])
                plt.scatter(Y[i, 0], Y[i, 1], 40, 'r')
                plt.axis('equal')
                ax = plt.gca()
                ax.set_facecolor((0.15, 0.15, 0.15))
                ax.set_xticks([])
                ax.set_yticks([])
                plt.subplot(2, ncols, ncols+2)
                plt.scatter(Y[:, 2], Y[:, 3], 100, c=np.array([[0, 0, 0, 0]]))
                plt.scatter(Y[0:i+1, 2], Y[0:i+1, 3], 20, c=C2[0:i+1, :])
                plt.scatter(Y[i, 2], Y[i, 3], 40, 'r')
                plt.axis('equal')
                ax = plt.gca()
                ax.set_facecolor((0.15, 0.15, 0.15))
                ax.set_xticks([])
                ax.set_yticks([])
            if D.size > 0:
                plt.subplot(233)
                plt.imshow(D, cmap='magma_r', interpolation='none')
                fac = float(D.shape[0])/Y.shape[0]

                plt.plot([fac*i, fac*i], [0, D.shape[1]], 'c')
                plt.plot([0, D.shape[1]], [fac*i, fac*i], 'c')
                plt.xlim([0, D.shape[1]])
                plt.ylim([D.shape[1], 0])

            plt.savefig("%i.png"%idx, bbox_inches='tight')
            idx += 1
        # Set patches back to what they were
        self.recover_original_patches()