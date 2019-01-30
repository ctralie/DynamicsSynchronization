import numpy as np
import scipy.io as sio
import scipy.linalg as slinalg
import matplotlib.pyplot as plt
import sklearn.feature_extraction.image as skimage
from sklearn import manifold
import umap
import time
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
from ripser import ripser, plot_dgms
import sys
import warnings
sys.path.append("DREiMac")
from CircularCoordinates import CircularCoords
from DiffusionMaps import *
from LocalPCA import *

def makeVideo(Y, I, Xs, Ts, pd, colorvar, skip=20):
    c = plt.get_cmap('magma_r')
    C = c(np.array(np.round(255.0*colorvar/np.max(colorvar)), dtype=np.int32))
    C = C[:, 0:3]
    fig = plt.figure(figsize=(12, 6))
    idx = 0
    for i in range(0, Y.shape[0], skip):
        plt.clf()
        plt.subplot(121)
        plt.imshow(I, interpolation='none', aspect='auto', cmap='RdGy')
        x1, x2 = Xs[i], Xs[i]+pd[1]
        y1, y2 = Ts[i], Ts[i]+pd[0]
        plt.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], 'r')
        plt.xlabel("Space")
        plt.ylabel("Time")
        plt.title("[%i, %i] x [%i, %i]"%(x1, x2, y1, y2))
        plt.xlim([0, I.shape[1]])
        plt.ylim([I.shape[0], 0])

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

def do_umap_and_tda(pd, sub, nperm, patches, I, Xs, Ts, ts):
    ## Step 1: Do Umap
    n_components = 2
    n_neighbors = 4
    Y = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors).fit_transform(patches)


    print(patches.shape)
    res1 = CircularCoords(patches, nperm, cocycle_idx = [0])
    #res2 = CircularCoords(patches, nperm, cocycle_idx = [1])
    perm = res1["perm"]
    dgms = ripser(patches[perm, :], maxdim=1, coeff=41)["dgms"]
    dgms1 = dgms[1]


    plt.figure(figsize=(12, 12))
    plt.subplot(221)
    plt.imshow(I, interpolation='none', aspect='auto', cmap='RdGy', extent=(0, I.shape[1], ts[-1], ts[0]))
    x1, x2 = Xs[5], Xs[5]+pd[1]
    y1, y2 = ts[Ts[5]], ts[Ts[5]+pd[0]]
    plt.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], 'r')
    plt.xlabel("Space")
    plt.ylabel("Time")
    plt.title("Solution")

    plt.subplot(223)
    plt.scatter(Y[:, 0], Y[:, 1], 20, c=ts[Ts], cmap='magma_r')
    plt.colorbar()
    plt.axis('equal')
    plt.title("Umap By Time")


    plt.subplot(224)
    plt.scatter(Y[:, 0], Y[:, 1], 20, c=Xs, cmap='magma_r')
    plt.colorbar()
    plt.axis('equal')
    plt.title("Umap By Space")

    plt.subplot(222)
    plot_dgms(dgms)
    idx = np.argsort(dgms1[:, 0]-dgms1[:, 1])
    plt.text(dgms1[idx[0], 0], dgms1[idx[0], 1], "1")
    plt.text(dgms1[idx[1], 0], dgms1[idx[1], 1], "2")
    plt.title("Persistence Diagrams")

    """
    plt.subplot(235)
    plt.scatter(Xs, ts[Ts], 40, res1["thetas"], cmap="magma_r")
    plt.gca().invert_yaxis()
    plt.xlabel("Space")
    plt.ylabel("Time")
    plt.xlim([0, I.shape[1]])
    plt.ylim([ts[-1], ts[0]])
    plt.title("Cocycle 1")

    plt.subplot(236)
    plt.scatter(Xs, ts[Ts], 40, res2["thetas"], cmap="magma_r")
    plt.gca().invert_yaxis()
    plt.xlabel("Space")
    plt.ylabel("Time")
    plt.xlim([0, I.shape[1]])
    plt.ylim([ts[-1], ts[0]])
    plt.title("Cocycle 2")
    """
    plt.show()
    return Y

def doDiffusionMaps(DSqr, Xs, Ts, ts, dMaxSqrCoeff = 1.0):
    """
    Using https://github.com/jmbr/diffusion-maps
    """
    c = plt.get_cmap('magma_r')
    C1 = c(np.array(np.round(255.0*ts[Ts]/np.max(ts[Ts])), dtype=np.int32))
    C1 = C1[:, 0:3]
    C2 = c(np.array(np.round(255.0*Xs/np.max(Xs)), dtype=np.int32))
    C2 = C2[:, 0:3]
    
    t = dMaxSqrCoeff*np.max(DSqr)*0.001
    print("t = %g"%t)
    print("Doing diffusion maps on %i points"%(DSqr.shape[0]))
    tic = time.time()
    Y = getDiffusionMap(DSqr, t, distance_matrix=True, neigs=6, thresh=1e-10)
    Y = np.fliplr(Y)
    print("Elapsed Time: %.3g"%(time.time()-tic))
    Y = Y[:, 1::]
    plt.imshow(Y, aspect='auto')
    plt.show()

    plt.figure(figsize=(6, 6))
    """
    ax = plt.gcf().add_subplot(121, projection='3d')
    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=C1)
    plt.title("Diffusion Maps By Time")
    """

    ax = plt.gcf().add_subplot(111, projection='3d')
    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=C2)
    plt.title("Diffusion Maps By Space")
    plt.show()

    return Y














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
        Spatial indices of the left of each patch into I
    Ts: ndarray(N)
        Time indices of the top of each patch into I
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
    
    def makeObservations(self, pd, sub, tidx_max, f = lambda x: x):
        """
        Create an observation 
        Parameters
        ----------
        pd: tuple(int, int)
            The dimensions of each patch (height, width)
        sub: tuple(int, int)
            The factor by which to subsample the patches across each dimension
        f: function ndarray->ndarray
            A pointwise homeomorphism to apply to pixels in the observation function
        tidx_max: int
            Maximum time index to include in any observation window
        """
        I = self.I[0:tidx_max, :]
        I = np.concatenate((I, I[:, 0:pd[1]]), 1) # Do periodic padding
        M, N = I.shape[0], I.shape[1]
        patches = skimage.extract_patches_2d(I, pd)
        patches = np.reshape(patches, (M-pd[0]+1, N-pd[1]+1, pd[0], pd[1]))
        # Index by spatial coordinate and by time
        Xs, Ts = np.meshgrid(np.arange(patches.shape[1]), np.arange(patches.shape[0]))
        # Subsample patches
        patches = patches[0::sub[0], 0::sub[1], :, :]
        Xs = Xs[0::sub[0], 0::sub[1]]
        Ts = Ts[0::sub[0], 0::sub[1]]
        Xs = Xs.flatten()
        Ts = Ts.flatten()
        self.Xs = Xs
        self.Ts = Ts
        self.patches = np.reshape(patches, (patches.shape[0]*patches.shape[1], pd[0]*pd[1]))
        self.pd = pd
        self.f = f
    
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
        I = self.I
        x = np.arange(I.shape[1])
        x = x[None, :]*np.ones((3, 1))
        x[0, :] -= I.shape[1]
        x[2, :] += I.shape[1]
        x = x.flatten()
        t = np.arange(I.shape[0])
        I = np.concatenate((I, I, I), 1) #Periodic boundary conditions in space
        f_interp = interpolate.RectBivariateSpline(t, x, I)

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
            thetas = np.pi*np.random.rand(n_points)
            xs = x0 + rs*np.cos(thetas) # Left of each patch sample
            ts = t0 + rs*np.sin(thetas) # Top of each patch sample
            xs = xs[:, None] + pdx[None, :]
            ts = ts[:, None] + pdt[None, :]
            Y = self.f(f_interp(ts.flatten(), xs.flatten(), grid=False))
            Y = np.reshape(Y, (n_points, pdx.size)) - y[None, :]
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


def testKS_NLDM(pd = (150, 1), sub=(2, 1), dMaxSqrCoeff = 1.0, skip=15, nperm = 600):
    """
    Test a nonlinear dimension reduction of the Kuramoto Sivashinsky Equation
    torus attractor
    Parameters
    ----------
    pd: tuple(int, int)
        The dimensions of each patch
    sub: int
        The factor by which to subsample the patches
    nperm: int
        Number of points to take in a greedy permutation
    """
    ks = KSSimulation()
    ks.makeObservations(pd, sub, -50)
    #Y = do_umap_and_tda(pd, sub, nperm, ks.patches, ks.I, ks.Xs, ks.Ts, ks.ts)

    D = np.sum(ks.patches**2, 1)[:, None]
    DSqr = D + D.T - 2*ks.patches.dot(ks.patches.T)
    Y = doDiffusionMaps(DSqr, ks.Xs, ks.Ts, ks.ts, dMaxSqrCoeff)
    makeVideo(Y, ks.I, ks.Xs, ks.Ts, pd, ks.Ts, skip)




def testKS_Mahalanobis(pd = (15, 15), sub=(2, 4)):
    ks = KSSimulation()
    ks.makeObservations(pd, sub, -50)
    DSqr = ks.getMahalanobisDists(delta=3, n_points=100, d=2)
    Y = doDiffusionMaps(DSqr, ks.Xs, ks.Ts, ks.ts, dMaxSqrCoeff = 12.0)
    sio.savemat("Y.mat", {"Y":Y})
    perm, lambdas = getGreedyPerm(Y, 600)
    plt.figure()
    dgms = ripser(Y[perm, :], maxdim=2)["dgms"]
    plot_dgms(dgms, show=False)
    plt.show()
    makeVideo(Y, ks.I, ks.Xs, ks.Ts, pd, ks.Ts, skip=2)


if __name__ == '__main__':
    #testKS_NLDM(pd = (50, 15), sub=(1, 4), dMaxSqrCoeff=0.07)
    testKS_Mahalanobis(pd = (50, 8), sub=(1, 4))