import numpy as np
import scipy.io as sio
import scipy.linalg as slinalg
from skimage.transform import resize
from scipy.ndimage.filters import gaussian_filter1d as gf1d
import matplotlib.pyplot as plt
from PDE2D import *
from Mahalanobis import *
from SyntheticExamples import FlatTorusIdeal

class KSSimulation(PDE2D):
    def __init__(self, co_rotating = False, scale=(1.0, 1.0)):
        PDE2D.__init__(self)
        res = sio.loadmat("KS.mat")
        self.I = res["data"]
        self.c = res["c"] # Traveling wave factor
        M, N = self.I.shape[0], self.I.shape[1]
        if co_rotating:
            self.make_corotating()
        self.I = resize(self.I, (M*scale[0], N*scale[1]), anti_aliasing=True, mode='reflect')
        self.ts = np.linspace(res["tmin"].flatten(), res["tmax"].flatten(), M)
        self.xs = np.linspace(0, 2*np.pi, self.I.shape[1]+1)[0:self.I.shape[1]]
    
    def crop(self, t0, t1, x0, x1):
        self.I = self.I[t0:t1, x0:x1]
        self.ts = self.ts[t0:t1]
        self.xs = self.xs[x0:x1]
    
    def make_corotating(self, do_plot=False):
        """
        Warp into a co-rotating frame by circularly shifting each time slice
        by an appropriate amount
        """
        ratio = self.c
        INew = np.array(self.I)
        N = self.I.shape[1]
        for i in range(1, self.I.shape[0]):
            pix = np.arange(N)
            pix = np.concatenate((pix-N, pix, pix+N))
            x = self.I[i, :]
            x = np.concatenate((x, x, x))
            spl = InterpolatedUnivariateSpline(pix, x)
            INew[i, :] = spl((np.arange(N) + i*ratio) % N)
        self.I = INew

    def drawSolutionImage(self, time_extent = True, inc = 0.02):
        if time_extent:
            from matplotlib.ticker import FormatStrFormatter
            m = np.max(np.abs(self.I))
            I = np.array(self.I)
            I = np.concatenate((I, I, I), 1)
            plt.imshow(I, extent = (-self.I.shape[1], 2*self.I.shape[1], self.I.shape[0], 0), cmap='RdGy', vmin=-m, vmax=m)
            d = self.I.shape[1]/2
            plt.xticks([-2*d, -d, 0, d, 2*d, 3*d, 4*d], ['$-2\\pi$', '$-\\pi$', '0', '$\\pi$', '$2 \\pi$', '$3 \\pi$', '$4 \\pi$'])
            plt.xlim([-0.5*d, d*2.7])
            ts = self.ts
            ms = inc/(ts[1]-ts[0])
            n = np.ceil(ts.size/ms)
            idx = np.arange(n)*ms
            ts = ['%.3g'%t for t in ts[np.array(idx, dtype=int)]]
            plt.yticks(idx, ts)
            plt.xlabel("Space (Radians)")
            plt.ylabel("Time (Seconds)")
        else:
            plt.imshow(self.I, interpolation='nearest', aspect='auto', cmap='RdGy')
    
    def makeTimeSeriesVideo(self):
        plt.figure(figsize=(12, 4))
        m = np.max(np.abs(self.I))
        for i, idx in enumerate(range(0, self.I.shape[0], 10)):
            plt.clf()
            plt.subplot(121)
            self.drawSolutionImage(True)
            plt.colorbar()
            plt.plot([0, 2*np.pi], [self.ts[idx]]*2)
            plt.xlabel("Space (Radians)")
            plt.ylabel("Time (Seconds)")
            plt.title("Spacetime Grid")
            plt.subplot(122)
            plt.plot(self.xs, self.I[idx, :])
            plt.ylim([-m*1.1, m*1.1])
            plt.xlabel("Space (Radians)")
            plt.title("Timeslice at %g Seconds"%self.ts[idx])
            plt.savefig("%i.png"%i, bbox_inches='tight')



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

def testKS_VerticalOnly():
    cmap = 'magma_r'
    dMaxSqr = 1
    fac = 0.5
    ks = KSSimulation(co_rotating=True, scale=(fac, fac/2))
    ks.I = ks.I[0:200, :]
    print("Shape = (%i, %i)"%ks.I.shape)
    dim = 64
    #ks.Ts = np.linspace(dim/2, ks.I.shape[0]-dim/2, 200)
    #ks.Xs = ks.I.shape[1]/2*np.ones_like(ks.Ts)
    ks.Xs = np.linspace(0, ks.I.shape[1], 200)
    ks.Ts = ks.I.shape[0]/2*np.ones_like(ks.Xs)
    ks.thetas = (np.pi/4)*np.ones_like(ks.Ts)
    ks.pca = None
    ks.pd = (dim, dim)
    ks.completeObservations()
    Xs, Ts = ks.Xs, ks.Ts

    D = np.sum(ks.patches**2, 1)[:, None]
    DSqr = D + D.T - 2*ks.patches.dot(ks.patches.T)
    DSqr[DSqr < 0] = 0
    mask = np.ones_like(DSqr)
    D = np.sqrt(DSqr)
    D[mask == 0] = np.inf

    Y = doDiffusionMaps(DSqr, Xs, dMaxSqr, do_plot=False, mask=mask, neigs=8)
    

    fig = plt.figure(figsize=(18, 6))
    if Y.shape[1] > 2:
        ax = fig.add_subplot(131, projection='3d')
        ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=Xs, cmap=cmap)
    else:
        plt.subplot(131)
        plt.scatter(Y[:, 0], Y[:, 1], c=Xs, cmap=cmap)
    plt.title("X")
    if Y.shape[1] > 2:
        ax = fig.add_subplot(132, projection='3d')
        ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=Ts, cmap=cmap)
    else:
        plt.subplot(132)
        plt.scatter(Y[:, 0], Y[:, 1], c=Ts, cmap=cmap)
    plt.title("T")
    plt.subplot(133)
    plt.imshow(largeimg(D, np.ones_like(D)), aspect='auto', cmap=cmap)
    plt.show()

    ks.makeVideo(Y, D, skip=1, cmap=cmap, colorvar=Xs)


def testKS_Alignment():
    from ICP import doICP_PDE2D
    ## Step 1: Setup stretched KS
    # Setup to go through roughly 3 periods so it's easy to 
    # come up with correspondences
    fac = 0.5
    ks = KSSimulation(co_rotating=False, scale=(fac*7, fac/2))
    ks.crop(0, 170, 0, ks.I.shape[1])

    ## Step 2: Run Mahalanobis on (rotated) patches
    """
    noisefac = 0.01
    ks.I += noisefac*np.max(np.abs(ks.I))*np.random.randn(ks.I.shape[0], ks.I.shape[1])
    Yks = testMahalanobis_PDE2D(ks, pd=(32, 32), nsamples=5000, \
                    dMaxSqr=100, delta=2, rank=2, maxeigs=85, jacfac=0.75,\
                    periodic=True, rotate=True, use_rotinvariant=True, \
                    do_mahalanobis=True, \
                    precomputed_samples=None, pca_dim=120, do_plot=True, do_tda=True, do_video=False)
    """

    #"""
    noisefac = 0.01
    ks.I += noisefac*np.max(np.abs(ks.I))*np.random.randn(ks.I.shape[0], ks.I.shape[1])
    Yks = testMahalanobis_PDE2D(ks, pd=(45, 45), nsamples=4000, \
                    dMaxSqr=1000, delta=2, rank=2, maxeigs=85, jacfac=0.75,\
                    periodic=True, rotate=True, use_rotinvariant=True, \
                    do_mahalanobis=True, \
                    precomputed_samples=None, pca_dim=120, do_plot=True, do_tda=True, do_video=False)
    #"""

    ## Step 3: Run alignment to ideal torus
    ft = FlatTorusIdeal(60, 60, Yks.shape[1])
    Yft = ft.Y
    doICP_PDE2D(ks, Yks[:, 0:4], ft, Yft[:, 0:4], initial_guesses=10, do_plot=True)

def plotKS():
    fac = 0.5
    ks = KSSimulation(co_rotating=False, scale=(fac*7, fac/2))
    ks.crop(0, 170, 0, ks.I.shape[1])
    ks.drawSolutionImage()
    plt.show()
    
    #ks.makeObservations((64, 20), 100, rotate=True)
    #ks.plotPatches()
    #ks.makeTimeSeriesVideo()

if __name__ == '__main__':
    #testKS_VerticalOnly()
    #testKS_Variations(ks)
    #testKS_Rotations(ks)
    #testKS_Alignment()
    plotKS()