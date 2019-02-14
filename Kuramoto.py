import numpy as np
import scipy.io as sio
import scipy.linalg as slinalg
import matplotlib.pyplot as plt
import sklearn.feature_extraction.image as skimage
import time
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
from ripser import ripser, plot_dgms
import torch
from kymatio import Scattering2D
import sys
import warnings
sys.path.append("DREiMac")
from CircularCoordinates import CircularCoords
from DiffusionMaps import *
from LocalPCA import *
from PatchDescriptors import *

def doDiffusionMaps(DSqr, Xs, dMaxSqrCoeff = 1.0, do_plot = True):
    c = plt.get_cmap('magma_r')
    C2 = c(np.array(np.round(255.0*Xs/np.max(Xs)), dtype=np.int32))
    C2 = C2[:, 0:3]
    
    t = dMaxSqrCoeff*np.max(DSqr)*0.001
    print("t = %g"%t)
    print("Doing diffusion maps on %i points"%(DSqr.shape[0]))
    tic = time.time()
    Y = getDiffusionMap(DSqr, t, distance_matrix=True, neigs=4, thresh=1e-10)
    Y = np.fliplr(Y)
    print("Elapsed Time: %.3g"%(time.time()-tic))
    Y = Y[:, 1::]

    if do_plot:
        ax = plt.gcf().add_subplot(111, projection='3d')
        ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=C2)
        plt.title("Diffusion Maps By Space")
    return Y


def even_interval(k):
    """
    Return unit samples equally spaced around 0
    """
    if k%2 == 0:
        n = k/2
        return 0.5+np.arange(k)-n
    n = (k-1)/2
    return np.arange(k)-n


class KSSimulation(object):
    """
    Attributes
    ----------
    I: ndarray(T, S)
        The full Kuramoto Sivashinsky spacetime grid
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
        res = sio.loadmat("KS.mat")
        I = res["data"]
        self.I = I
        self.ts = np.linspace(res["tmin"].flatten(), res["tmax"].flatten(), I.shape[0])
    
    def drawSolutionImage(self, time_extent = False):
        if time_extent:
            plt.imshow(self.I, interpolation='none', aspect='auto', \
                        cmap='RdGy', extent=(0, self.I.shape[1], \
                        self.ts[-1], self.ts[0]))
        else:
            plt.imshow(self.I, interpolation='none', aspect='auto', cmap='RdGy')

    def getInterpolator(self, periodic = True):
        """
        Create a 2D rect bivariate spline interpolator to sample
        from the solution set with interpolation
        """
        ## Step 1: Setup interpolator
        I = self.I
        x = np.arange(I.shape[1])
        t = np.arange(I.shape[0])
        if periodic:
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

    def makeObservations(self, pd, nsamples, uniform=True, rotate = False, \
                            f_pointwise = lambda x: x, f_patch = lambda x: x):
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
        rotate: boolean
            Whether to randomly rotate the patches
        f: function ndarray->ndarray
            A pointwise homeomorphism to apply to pixels in the observation function
        """
        f_interp = self.getInterpolator(periodic=True)
        # Make sure a rotated patch is within the time range
        # (we usually don't have to worry about )
        rotstr = ""
        if rotate:
            rotstr = " rotated"
            r = np.sqrt((pd[0]/2)**2 + (pd[1]/2)**2)
        else:
            r = pd[0]/2.0
        print("Making %s%s observations of dimension %s..."%(nsamples, rotstr, pd))
        self.pd = pd
        self.f_pointwise = f_pointwise
        self.f_patch = f_patch
        if isinstance(nsamples, tuple):
            M, N = nsamples
            x = np.linspace(0, self.I.shape[1], N)
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
                self.Xs = Y[:, 1]*self.I.shape[1]
            else:
                self.Ts = r+np.random.rand(N)*(self.I.shape[0]-2*r)
                self.Xs = np.random.rand(N)*self.I.shape[1]
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
        for i, (theta, tc, xc) in enumerate(zip(self.thetas, self.Ts, self.Xs)):
            c, s = np.cos(theta), np.sin(theta)
            xs[i, :] = c*pdx - s*pdt + xc
            ts[i, :] = s*pdx + c*pdt + tc
        
        # Use interpolator to sample coordinates for all patches
        patches = (f_interp(ts.flatten(), xs.flatten(), grid=False))
        patches = np.reshape(patches, ts.shape)
        self.patches = f_patch(f_pointwise(patches))

    def getMahalanobisDists(self, delta, n_points, d = -10, maxeig = 10):
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
        """
        if maxeig < d:
            warnings.warn("maxeig = %i < d = %i"%(maxeig, d))
        dim = self.patches.shape[1] #The dimension of each patch
        maxeig = min(maxeig, d)
        maxeig = min(maxeig, dim)
        ## Step 1: Setup interpolator
        f_interp = self.getInterpolator(periodic=True)

        ## Step 2: For each patch, sample near patches
        ## and compute covariance matrix
        pdx, pdt = np.meshgrid(np.arange(self.pd[1]), np.arange(self.pd[0]))
        pdx = pdx.flatten()
        pdt = pdt.flatten()
        N = self.patches.shape[0]

        print("Computing Jacobians...")
        tic = time.time()
        #"""
        ws = np.zeros((N, maxeig))
        vs = np.zeros((N, dim, maxeig))
        for i in range(N):
            y = self.patches[i, :]
            if i%100 == 0:
                print("%i of %i"%(i, N))
            x0 = self.Xs[i]
            t0 = self.Ts[i]
            rs = delta*np.sqrt(np.random.rand(n_points))
            thetas = 2*np.pi*np.random.rand(n_points)
            xs = x0 + rs*np.cos(thetas) # Left of each patch sample
            ts = t0 + rs*np.sin(thetas) # Top of each patch sample
            xs = xs[:, None] + pdx[None, :]
            ts = ts[:, None] + pdt[None, :]
            Y = self.f(f_interp(ts.flatten(), xs.flatten(), grid=False))
            Y = np.reshape(Y, (n_points, pdx.size)) - y[None, :] # Center samples
            C = (Y.T).dot(Y)
            w, v = slinalg.eigh(C, eigvals=(C.shape[1]-maxeig, C.shape[1]-1))
            # Put largest eigenvectors first
            w = w[::-1]
            v = np.fliplr(v)
            ws[i, :] = w
            vs[i, :, :] = v
        sio.savemat("vs.mat", {"vs":vs, "ws":ws})
        #"""
        #"""
        res = sio.loadmat("vs.mat")
        vs, ws = res["vs"], res["ws"]
        #"""
        print("Elapsed Time: %.3g"%(time.time()-tic))
        if d < 0:
            # Estimate dimension using eigengaps
            ratios = np.median(ws[:, 0:-1]/ws[:, 1::], 0)
            d = np.argmax(np.sign(ratios+d)) + 1

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
            Dk_diag = np.diag(Dk)
            wk = ws[:, k]
            gamma += (Dk_diag[:, None] - Dk.T)**2/wk[:, None]
            gamma += (Dk - Dk_diag[None, :])**2/wk[None, :]
        gamma *= 0.5*(delta**2)/(maxeig+2)
        print("Elapsed Time: %.3g"%(time.time()-tic))
        return gamma


    
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

    def makeVideo(self, Y, skip=20):
        """
        Make a video given a nonlinear dimension reduction, which
        is assumed to be parallel to the patches
        """
        colorvar = self.Xs
        c = plt.get_cmap('magma_r')
        C = c(np.array(np.round(255.0*colorvar/np.max(colorvar)), dtype=np.int32))
        C = C[:, 0:3]
        fig = plt.figure(figsize=(12, 6))
        I, Xs, Ts = self.I, self.Xs, self.Ts

        idx = 0
        for i in range(0, Y.shape[0], skip):
            plt.clf()
            plt.subplot(121)
            plt.imshow(I, interpolation='none', aspect='auto', cmap='RdGy')
            self.plotPatchBoundary(i)
            plt.xlabel("Space")
            plt.ylabel("Time")
            plt.axis('equal')
            plt.xlim(0, I.shape[1])
            plt.ylim(I.shape[0], 0)

            if Y.shape[1] == 3:
                ax = plt.gcf().add_subplot(122, projection='3d')
                ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=np.array([[0, 0, 0, 0]]))
                ax.scatter(Y[0:i+1, 0], Y[0:i+1, 1], Y[0:i+1, 2], c=C[0:i+1, :])
                ax.scatter(Y[i, 0], Y[i, 1], Y[i, 2], 'r')
            else:
                plt.subplot(122)
                plt.scatter(Y[:, 0], Y[:, 1], 100, c=np.array([[0, 0, 0, 0]]))
                plt.scatter(Y[0:i+1, 0], Y[0:i+1, 1], 20, c=C[0:i+1, :])
                plt.scatter(Y[i, 0], Y[i, 1], 40, 'r')
                plt.axis('equal')
                ax = plt.gca()
                ax.set_facecolor((0.15, 0.15, 0.15))
                ax.set_xticks([])
                ax.set_yticks([])
            plt.savefig("%i.png"%idx)
            idx += 1





def testKS_NLDM(pd = (150, 1), nsamples=1000, dMaxSqrCoeff = 1.0, skip=15, nperm = 600):
    """
    Test a nonlinear dimension reduction of the Kuramoto Sivashinsky Equation
    torus attractor
    Parameters
    ----------
    pd: tuple(int, int)
        The dimensions of each patch
    nsamples: int or tuple(int, int)
        The number of patches to sample, or the dimension of the 
        uniform grid from which to sample patches
    nperm: int
        Number of points to take in a greedy permutation
    """
    ks = KSSimulation()
    f_patch = lambda patches: get_derivative_shells(patches, pd, orders=[0, 1], n_shells=50)
    #f_patch = lambda patches: get_scattering(patches, pd, rotinvariant=False)
    ks.makeObservations(pd, nsamples, rotate=True, f_patch=f_patch)
    #ks.makeObservationsIntegerGrid(pd, (2, 2))

    D = np.sum(ks.patches**2, 1)[:, None]
    DSqr = D + D.T - 2*ks.patches.dot(ks.patches.T)
    print(np.max(DSqr))
    Y = doDiffusionMaps(DSqr, ks.Xs, dMaxSqrCoeff)
    plt.show()
    ks.makeVideo(Y[:, 0:2], skip)



def testKS_Mahalanobis(pd = (15, 15), NSamples = 10000):
    ks = KSSimulation()
    ks.makeObservations(pd, NSamples)
    DSqr = ks.getMahalanobisDists(delta=3, n_points=100, d=2)
    Y = doDiffusionMaps(DSqr, ks.Xs, dMaxSqrCoeff = 10.0)
    sio.savemat("Y.mat", {"Y":Y})
    perm, lambdas = getGreedyPerm(Y, 600)
    plt.figure()
    dgms = ripser(Y[perm, :], maxdim=2)["dgms"]
    plot_dgms(dgms, show=False)
    plt.show()
    ks.makeVideo(Y[:, 0:2], skip=15)


def testKS_Variations():
    """
    Vary window lengths and diffusion parameters
    """
    ks = KSSimulation()
    for N in [5000, 10000, 15000]:
        for pd in [(150, 1)]:#[(200, 1), (150, 1), (50, 50), (1, 150), (40, 40)]:
            #ks.makeObservationsIntegerGrid(pd, (2, 2))
            ks.makeObservations(pd, N)
            D = np.sum(ks.patches**2, 1)[:, None]
            DSqr = D + D.T - 2*ks.patches.dot(ks.patches.T)
            for dMaxSqrCoeff in [0.5]:#np.array([0.3, 0.4, 0.5, 1.0]):
                plt.clf()
                Y = doDiffusionMaps(DSqr, ks.Xs, dMaxSqrCoeff, True)
                plt.title("%i x %i Patches, $\\epsilon=%.3g \\max(D^2) 10^{-3}$"%(pd[0], pd[1], dMaxSqrCoeff))
                plt.savefig("%i_%i_%.3g_%i.png"%(pd[0], pd[1], dMaxSqrCoeff, N))
                ks.makeVideo(Y, skip=20)

def testKS_Rotations():
    ks = KSSimulation()
    ks.makeObservations((80, 40), (20, 20), rotate=False)
    
    ks.plotPatches(False)
    plt.show()
    ks.plotPatches()

if __name__ == '__main__':
    testKS_NLDM(pd = (64, 64), nsamples=(94, 201), dMaxSqrCoeff=10, skip=1)
    #testKS_Mahalanobis(pd = (50, 15), sub=(1, 4))
    #testKS_Variations()
    #testKS_Rotations()