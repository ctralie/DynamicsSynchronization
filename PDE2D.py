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


def even_interval(k):
    """
    Return unit samples equally spaced around 0
    """
    if k%2 == 0:
        n = k/2
        return 0.5+np.arange(k)-n
    n = (k-1)/2
    return np.arange(k)-n


class PDE2D(object):
    """
    Attributes
    ----------
    I: ndarray(T, S)
        The full 2D PDE spacetime grid
    ts: ndarray(T)
        Times corresponding to each row of I
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
    f: function ndarray->ndarray
        A pointwise homeomorphism to apply to pixels in the observation function
    """
    def __init__(self):
        self.I = np.array([[]])
        self.periodic = True

    def drawSolutionImage(self):
        plt.imshow(self.I, interpolation='none', aspect='auto', cmap='RdGy')

    def getInterpolator(self):
        """
        Create a 2D rect bivariate spline interpolator to sample
        from the solution set with interpolation
        """
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
        return interpolate.RectBivariateSpline(t, x, I)

    def makeObservationsIntegerGrid(self, pd, sub, tidx_max = None, \
                                    f_pointwise = lambda x: x, f_patch = lambda x: x):
        """
        Make observations without rotation on a regular grid  
        Parameters
        ----------
        pd: tuple(int, int)
            The dimensions of each patch (height, width)
        sub: tuple(int, int)
            The factor by which to subsample the patches across each dimension
        tidx_max: int
            Maximum time index to include in any observation window
        f_pointwise: function ndarray->ndarray
            A pointwise homeomorphism to apply to pixels in the observation function
        f_patch: funtion: ndarray(n_patches, n_pixels) -> ndarray(n_patches, n_features)
            A function to apply to all patches
        """
        print("Making observations on integer grid, pd = %s, sub = %s..."%(pd, sub))
        I = self.I
        if tidx_max:
            I = I[0:tidx_max, :]
        I = np.concatenate((I, I[:, 0:pd[1]]), 1) # Do periodic padding
        M, N = I.shape[0], I.shape[1]
        patches = skimage.extract_patches_2d(I, pd)
        patches = np.reshape(patches, (M-pd[0]+1, N-pd[1]+1, pd[0], pd[1]))
        # Index by spatial coordinate and by time
        Xs, Ts = np.meshgrid(np.arange(patches.shape[1]), np.arange(patches.shape[0]))
        Xs = np.array(Xs, dtype=float)
        Ts = np.array(Ts, dtype=float)
        Xs += float(pd[1])/2.0
        Ts += float(pd[0])/2.0
        # Subsample patches
        patches = patches[0::sub[0], 0::sub[1], :, :]
        patches = f_patch(f_pointwise(patches))
        Xs = Xs[0::sub[0], 0::sub[1]]
        Ts = Ts[0::sub[0], 0::sub[1]]
        Xs = Xs.flatten()
        Ts = Ts.flatten()
        self.Xs = Xs
        self.Ts = Ts
        self.thetas = np.zeros_like(Xs)
        self.patches = np.reshape(patches, (patches.shape[0]*patches.shape[1], pd[0]*pd[1]))
        self.pd = pd
        self.f_pointwise = f_pointwise
        self.f_patch = f_patch
        self.rotate_patches = False

    def makeObservations(self, pd, nsamples, uniform=True, periodic=True, rotate = False, \
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
            Whether to enforce periodic boundary conditions
        rotate: boolean
            Whether to randomly rotate the patches
        buff: float
            A buffer to include around the edges of the patches 
            that are sampled.  If periodic is true, only use
            this buffer in the vertical direction
        f_pointwise: function ndarray->ndarray
            A pointwise homeomorphism to apply to pixels in the observation function
        """
        self.periodic = periodic
        f_interp = self.getInterpolator()
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
        print("Making %s%s observations of dimension %s..."%(nsamples, rotstr, pd))
        self.pd = pd
        self.f_pointwise = f_pointwise
        self.f_patch = f_patch
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
                self.Ts = r+np.random.rand(N)*(self.I.shape[0]-2*r)
                if periodic:
                    self.Xs = np.random.rand(N)*self.I.shape[1]
                else:
                    self.Xs = r+np.random.rand(N)*(self.I.shape[1]-2*r)
        if rotate:
            self.thetas = np.random.rand(self.Xs.size)*2*np.pi
        else:
            self.thetas = np.zeros_like(self.Xs)

        # Now sample all patches
        pdx, pdt = np.meshgrid(even_interval(pd[1]), -even_interval(pd[0]))
        pdx = pdx.flatten()
        pdt = pdt.flatten()
        ts = np.zeros((self.Xs.size, pdt.size))
        xs = np.zeros((self.Xs.size, pdx.size))

        # Setup all coordinate locations to sample
        cs, ss = np.cos(self.thetas), np.sin(self.thetas)
        xs = cs[:, None]*pdx[None, :] - ss[:, None]*pdt[None, :] + self.Xs[:, None]
        ts = ss[:, None]*pdx[None, :] + cs[:, None]*pdt[None, :] + self.Ts[:, None]
        
        # Use interpolator to sample coordinates for all patches
        patches = (f_interp(ts.flatten(), xs.flatten(), grid=False))
        patches = np.reshape(patches, ts.shape)
        self.patches = f_patch(f_pointwise(patches))

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
        if maxeig < d:
            warnings.warn("maxeig = %i < d = %i"%(maxeig, d))
        dim = self.patches.shape[1] #The dimension of each patch
        maxeig = min(maxeig, d)
        maxeig = min(maxeig, dim)
        
        ## Step 1: Setup interpolator and observation functions
        f_interp = self.getInterpolator()
        f_pointwise = self.f_pointwise
        f_patch = self.f_patch

        ## Step 2: For each patch, sample near patches
        ## and compute covariance matrix
        pdx, pdt = np.meshgrid(even_interval(self.pd[1]), -even_interval(self.pd[0]))
        pdx = pdx.flatten()
        pdt = pdt.flatten()
        N = self.patches.shape[0]

        print("Computing Jacobians...")
        tic = time.time()
        ws = np.zeros((N, maxeig))
        vs = np.zeros((N, dim, maxeig))
        for i in range(N):
            y = self.patches[i, :]
            if i%100 == 0:
                print("%i of %i"%(i, N))
            x0 = self.Xs[i]
            t0 = self.Ts[i]
            # Sample centers of each neighboring patch
            # in a circle around the original patch
            thetasc = 2*np.pi*np.random.rand(n_points)
            xc = x0 + delta*np.cos(thetasc)
            tc = t0 + delta*np.sin(thetasc)
            # Randomly rotate each patch
            thetasorient = np.zeros(n_points)
            if self.rotate_patches:
                thetasorient = 2*np.pi*np.random.rand(n_points)
            cs = np.cos(thetasorient)
            ss = np.sin(thetasorient)
            xs = cs[:, None]*pdx[None, :] - ss[:, None]*pdt[None, :] + xc[:, None]
            ts = ss[:, None]*pdx[None, :] + cs[:, None]*pdt[None, :] + tc[:, None]
            patches = f_interp(ts.flatten(), xs.flatten(), grid=False)
            patches = np.reshape(patches, (n_points, pdx.size))
            patches = f_patch(f_pointwise(patches))
            Y = patches - y[None, :] # Center samples
            C = (Y.T).dot(Y)
            w, v = slinalg.eigh(C, eigvals=(C.shape[1]-maxeig, C.shape[1]-1))
            # Put largest eigenvectors first
            w = w[::-1]
            v = np.fliplr(v)
            ws[i, :] = w
            vs[i, :, :] = v
        print("Elapsed Time: %.3g"%(time.time()-tic))
        # Estimate dimension using eigengaps
        if not (d == 1):
            ratios = np.median(ws[:, 0:-1]/ws[:, 1::], 0)
            d_est = np.argmax(np.sign(ratios+d)) + 1
            print("d_est = %i"%d_est)
            if d < 0:
                d = d_est

        ## Step 3: Compute squared Mahalanobis distance between
        ## all pairs of points
        gamma = np.zeros((N, N))
        #Create a matrix whose ith row and jth column contains
        #the dot product of patch i and eigenvector vjk
        D = np.zeros((N, N, maxeig))
        for k in range(maxeig):
            D[:, :, k] = self.patches.dot(vs[:, :, k].T)
        print("Computing Mahalanobis Distances...")
        tic = time.time()
        for k in range(maxeig):
            Dk = D[:, :, k]
            wk = ws[:, k]
            Dk_diag = np.diag(Dk)
            gamma += (Dk_diag[:, None] - Dk.T)**2/wk[:, None]
            gamma += (Dk - Dk_diag[None, :])**2/wk[None, :]
        gamma *= 0.5*(delta**2)/(maxeig+2)
        print("Elapsed Time: %.3g"%(time.time()-tic))

        ## Step 4: Make distances infinity between points
        ## that are not mutual nearest neighbors in the original metric
        mask = np.ones((N, N))
        if kappa < 1:
            NNeighbs = int(np.floor(kappa*N))
            D = np.sum(self.patches**2, 1)[:, None]
            DSqr = D + D.T - 2*self.patches.dot(self.patches.T)
            J = np.argpartition(DSqr, NNeighbs, 1)[:, 0:NNeighbs]
            I = np.tile(np.arange(N)[:, None], (1, NNeighbs))
            V = np.ones(I.size)
            [I, J] = [I.flatten(), J.flatten()]
            mask = sparse.coo_matrix((V, (I, J)), shape=(N, N))
            mask = mask.toarray()
        return {'gamma':gamma, 'mask':mask}

    
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
        pdt = even_interval(self.pd[0])
        x0, x1 = pdx[0], pdx[-1]
        t0, t1 = pdt[0], pdt[-1]
        x = np.array([[x0, t0], [x0, t1], [x1, t1], [x1, t0], [x0, t0]])
        c, s = np.cos(self.thetas[i]), np.sin(self.thetas[i])
        R = np.array([[c, -s], [s, c]])
        xc = self.Xs[i]
        tc = self.Ts[i]
        x = (R.dot(x.T)).T + np.array([[xc, tc]])
        plt.plot(x[:, 0], x[:, 1], 'C0')
        ax = plt.gca()
        R[:, 0] *= self.pd[1]/2
        R[:, 1] *= self.pd[0]/2
        if draw_arrows:
            ax.arrow(xc, tc, R[0, 0], R[1, 0], head_width = 5, head_length = 3, fc = 'c', ec = 'c', width = 1)
            ax.arrow(xc, tc, R[0, 1], R[1, 1], head_width = 5, head_length = 3, fc = 'g', ec = 'g', width = 1)

    def plotPatches(self, save_frames = True):
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
                p = np.reshape(self.patches[i, :], self.pd)
                plt.imshow(p, interpolation='none', cmap='RdGy', vmin=np.min(self.I), vmax=np.max(self.I))
                plt.savefig("%i.png"%i, bbox_inches='tight')

    def makeVideo(self, Y, D = np.array([]), skip=20, cmap='magma_r'):
        """
        Make a video given a nonlinear dimension reduction, which
        is assumed to be indexed parallel to the patches
        """
        colorvar = self.Xs
        c = plt.get_cmap(cmap)
        C = c(np.array(np.round(255.0*colorvar/np.max(colorvar)), dtype=np.int32))
        C = C[:, 0:3]
        sprefix = 120
        if D.size > 0:
            fig = plt.figure(figsize=(18, 6))
            sprefix = 130
        else:
            fig = plt.figure(figsize=(12, 6))
        I, Xs, Ts = self.I, self.Xs, self.Ts

        idx = 0
        N = Y.shape[0]
        for i in range(0, N, skip):
            plt.clf()
            plt.subplot(sprefix+1)
            plt.imshow(I, interpolation='none', aspect='auto', cmap='RdGy')
            self.plotPatchBoundary(i)
            plt.xlabel("Space")
            plt.ylabel("Time")
            plt.axis('equal')
            plt.xlim(0, I.shape[1])
            plt.ylim(I.shape[0], 0)

            if Y.shape[1] == 3:
                ax = plt.gcf().add_subplot(sprefix+2, projection='3d')
                ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=np.array([[0, 0, 0, 0]]))
                ax.scatter(Y[0:i+1, 0], Y[0:i+1, 1], Y[0:i+1, 2], c=C[0:i+1, :])
                ax.scatter(Y[i, 0], Y[i, 1], Y[i, 2], 'r')
            else:
                plt.subplot(sprefix+2)
                plt.scatter(Y[:, 0], Y[:, 1], 100, c=np.array([[0, 0, 0, 0]]))
                plt.scatter(Y[0:i+1, 0], Y[0:i+1, 1], 20, c=C[0:i+1, :])
                plt.scatter(Y[i, 0], Y[i, 1], 40, 'r')
                plt.axis('equal')
                ax = plt.gca()
                ax.set_facecolor((0.15, 0.15, 0.15))
                ax.set_xticks([])
                ax.set_yticks([])
            if D.size > 0:
                plt.subplot(133)
                plt.imshow(D, cmap='magma_r', interpolation='none')

                plt.plot([i, i], [0, N], 'c')
                plt.plot([0, N], [i, i], 'c')
                plt.xlim([0, N])
                plt.ylim([N, 0])

            plt.savefig("%i.png"%idx)
            idx += 1




