import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import sklearn.feature_extraction.image as skimage
from sklearn import manifold
import umap
import time
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
from ripser import ripser, plot_dgms
import sys
sys.path.append("DREiMac")
from CircularCoordinates import CircularCoords
from DiffusionMaps import *
from LocalPCA import *

def makeVideo(Y, I, Xs, Ts, pd, colorvar):
    c = plt.get_cmap('magma_r')
    C = c(np.array(np.round(255.0*colorvar/np.max(colorvar)), dtype=np.int32))
    C = C[:, 0:3]
    fig = plt.figure(figsize=(12, 6))
    idx = 0
    for i in range(0, Y.shape[0], 20):
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

def doDiffusionMaps(patches, Xs, Ts, ts):
    """
    Using https://github.com/jmbr/diffusion-maps
    """
    c = plt.get_cmap('magma_r')
    C1 = c(np.array(np.round(255.0*ts[Ts]/np.max(ts[Ts])), dtype=np.int32))
    C1 = C1[:, 0:3]
    C2 = c(np.array(np.round(255.0*Xs/np.max(Xs)), dtype=np.int32))
    C2 = C2[:, 0:3]
    
    D = getSSM(patches)
    t = 0.4*np.max(D**2)*0.001
    print("t = %g"%t)
    print("Doing diffusion maps on %i points in %i dimensions..."%(patches.shape[0], patches.shape[1]))
    tic = time.time()
    Y = getDiffusionMap(patches, t, neigs=4, thresh=1e-10)
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
    
    def getJacobians(self, r, n_points):
        """
        To quote from the Singer/Coifman paper:
        "Suppose that we can identify which data points y (j ) belong to the 
                ellipsoid E y (i) ,δ and which reside outside it"
        Assume that an "equal space" ellipse is a circle of radius "r"
        in spacetime (since we are trying to find a map back to spacetime).  
        Sample points uniformly at random within that disc, and
        use observations centered at those points
        Parameters
        ----------
        r: float
            Spacetime radius from which to sample
        n_points: int
            Number of points to sample in the disc
        """
        ## Step 1: Setup interpolator
        I = self.I
        I = np.concatenate((I, I, I), 1) #Periodic boundary conditions in space
        x = np.arange(I.shape[1])
        x = x[None, :]*np.ones((3, 1))
        x[0, :] -= I.shape[1]
        x[2, :] += I.shape[1]
        t = np.arange(I.shape[0])
        f_interp = interpolate.interp2d(x, t, I, bounds_error=True)

        ## Step 2: For each patch, sample near patches
        ## and compute Jacobian
        pdx, pdt = np.meshgrid(np.arange(self.pd[1]), np.arange(self.pd[0]))
        pdx = pdx.flatten()
        pdt = pdt.flatten()
        for i in range(self.patches.shape[0]):
            x0 = self.Xs[i]
            t0 = self.Ts[i]
            rs = r*np.sqrt(np.random.rand(n_points))
            thetas = np.pi*np.random.rand(n_points)
            xs = x0 + rs*np.cos(thetas) # Left of each patch sample
            ts = t0 + rs*np.sin(thetas) # Top of each patch sample
            xs = xs[:, None] + pdx[None, :]
            ts = ts[:, None] + pdt[None, :]
            Y = self.f(f_interp(xs.flatten(), ts.flatten()))
            Y = np.reshape(Y, (n_points, pdx.size))

            ## TODO: Finish computing jacobian
            


        





def testKS_NLDM(pd = (150, 1), sub=(2, 1), nperm = 600):
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
    Y = doDiffusionMaps(ks.patches, ks.Xs, ks.Ts, ks.ts)
    makeVideo(Y, ks.I, ks.Xs, ks.Ts, pd, ks.Ts)





def testKS_Mahalanobis():
    res = sio.loadmat("KS.mat")
    I = res["data"]
    ts = np.linspace(res["tmin"].flatten(), res["tmax"].flatten(), I.shape[0])


if __name__ == '__main__':
    testKS_NLDM()
    #testKS_Mahalanobis()