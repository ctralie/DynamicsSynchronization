import numpy as np
import scipy.io as sio
import scipy.linalg as slinalg
from skimage.transform import resize
from scipy.ndimage.filters import gaussian_filter1d as gf1d
import matplotlib.pyplot as plt
from PDE2D import *
from PatchDescriptors import *
from Mahalanobis import *
from skimage.transform import hough_line, hough_line_peaks
from scipy.interpolate import InterpolatedUnivariateSpline

class KSSimulation(PDE2D):
    def __init__(self, co_rotating = False, scale=(1.0, 1.0)):
        PDE2D.__init__(self)
        res = sio.loadmat("KS.mat")
        self.I = res["data"]
        M, N = self.I.shape[0], self.I.shape[1]
        if co_rotating:
            self.make_corotating()
        self.I = resize(self.I, (M*scale[0], N*scale[1]), anti_aliasing=True, mode='reflect')
        print(self.I.shape)
        self.ts = np.linspace(res["tmin"].flatten(), res["tmax"].flatten(), M)
    
    def make_corotating(self, do_plot=False):
        """
        Warp into a co-rotating frame by circularly shifting each time slice
        by an appropriate amount.  The slope of the rotation is estimated
        using the Hough Transform of a gradient magnitude image
        """
        image = np.array(self.I)
        imx = gf1d(image, sigma=1, order=1, axis=0)
        imy = gf1d(image, sigma=1, order=1, axis=1)
        image = np.sqrt(imx**2 + imy**2)
        q = np.quantile(image.flatten(), 0.9)
        image[image < q] = 0
        h, theta, d = hough_line(image)
        angles = []
        for _, angle, _ in zip(*hough_line_peaks(h, theta, d)):
            angles.append(angle)
        angle = np.median(angles)
        INew = np.array(self.I)
        N = self.I.shape[1]
        for i in range(1, self.I.shape[0]):
            pix = np.arange(N)
            pix = np.concatenate((pix-N, pix, pix+N))
            x = self.I[i, :]
            x = np.concatenate((x, x, x))
            spl = InterpolatedUnivariateSpline(pix, x)
            INew[i, :] = spl((np.arange(N) - i*np.tan(angle)) % N)
        if do_plot:
            plt.figure(figsize=(12, 6))
            plt.subplot(121)
            plt.imshow(self.I, cmap='gray')
            dx = np.cos(angle)
            dy = np.sin(angle)
            plt.plot([0, -dy*INew.shape[1]], [0, dx*INew.shape[1]], 'r')
            plt.title("Original")
            plt.subplot(122)
            plt.imshow(INew)
            plt.title("Co-Rotating", cmap="gray")
            plt.show()
        self.I = INew

    def drawSolutionImage(self, time_extent = False):
        if time_extent:
            plt.imshow(self.I, interpolation='none', aspect='auto', \
                        cmap='RdGy', extent=(0, self.I.shape[1], \
                        self.ts[-1], self.ts[0]))
        else:
            plt.imshow(self.I, interpolation='none', aspect='auto', cmap='RdGy')



def testKS_Diffusion(pde, pd, nsamples, delta, rotate=False, use_rotinvariant = False, dMaxSqr=1.0, do_mahalanobis=False, rank=2, jacfac=1, maxeigs=10, noisefac=0.0, do_tda=False, make_video=True, cmap='magma_r'):
    Y = testMahalanobis_PDE2D(pde, pd=pd, nsamples=nsamples, \
                    dMaxSqr=dMaxSqr, delta=3, rank=2, maxeigs=maxeigs, jacfac=1,\
                    periodic=True, rotate=False, do_mahalanobis=do_mahalanobis, \
                    precomputed_samples=(200, 200), pca_dim=20, do_plot=True)
    if do_tda:
        from ripser import ripser
        from persim import plot_diagrams as plot_dgms
        perm, lambdas = getGreedyPerm(Y, 600)
        plt.figure()
        dgms = ripser(Y[perm, :], maxdim=2)["dgms"]
        plot_dgms(dgms, show=False)
        plt.show()


def testKS_Variations(ks):
    """
    Vary window lengths and diffusion parameters
    """
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

def testKS_Rotations(ks):
    ks.makeObservations((80, 40), (20, 20), rotate=True)
    ks.plotPatches(False)
    plt.show()
    ks.plotPatches()

if __name__ == '__main__':
    fac = 0.5
    ks = KSSimulation(co_rotating=True, scale=(fac, fac/2))

    """
    ks.makeObservations(pd=(32, 32), nsamples=(100, 100), periodic=True, buff=3, rotate=False)
    ks.compose_with_dimreduction(dim=3)
    X = ks.patches
    ax = plt.gcf().add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=ks.Ts)
    plt.show()
    """

    testKS_Diffusion(ks, pd = (32, 32), nsamples=(75, 50),
        dMaxSqr=1, delta=3, rank=2, maxeigs=10, jacfac=10, noisefac=0.001,
        rotate=False, use_rotinvariant=False, do_mahalanobis=True)
    #testKS_Variations(ks)
    #testKS_Rotations(ks)