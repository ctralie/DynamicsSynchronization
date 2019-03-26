import numpy as np
import scipy.io as sio
import scipy.linalg as slinalg
from skimage.transform import resize
import matplotlib.pyplot as plt
from PDE2D import *
from PatchDescriptors import *
from Mahalanobis import *

class KSSimulation(PDE2D):
    def __init__(self, scale=1.0):
        PDE2D.__init__(self)
        res = sio.loadmat("KS.mat")
        I = res["data"]
        if not (scale == 1.0):
            I = resize(I, (I.shape[0]*scale, I.shape[1]*scale), anti_aliasing=True, mode='reflect')
        print(I.shape)
        self.I = I
        self.ts = np.linspace(res["tmin"].flatten(), res["tmax"].flatten(), I.shape[0])
    
    def drawSolutionImage(self, time_extent = False):
        if time_extent:
            plt.imshow(self.I, interpolation='none', aspect='auto', \
                        cmap='RdGy', extent=(0, self.I.shape[1], \
                        self.ts[-1], self.ts[0]))
        else:
            plt.imshow(self.I, interpolation='none', aspect='auto', cmap='RdGy')



def testKS_Diffusion(pde, pd, nsamples, delta, rotate=False, use_rotinvariant = False, dMaxSqr=1.0, do_mahalanobis=False, rank=2, jacfac=1, maxeigs=10, do_tda=False, make_video=True, cmap='magma_r'):
    f_patch = lambda x: x
    if use_rotinvariant:
        f_patch = lambda patches: get_ftm2d_polar(patches, pd)

    noisefac = 0.001
    pde.I += noisefac*np.random.randn(pde.I.shape[0], pde.I.shape[1])
    pde.makeObservations(pd=pd, nsamples=nsamples, periodic=True, buff=delta, rotate=rotate, f_patch=f_patch)
    if not (type(nsamples) is tuple):
        pde.resort_byraster()
    Xs = pde.Xs
    Ts = pde.Ts
    N = Xs.size
    mask = np.ones((N, N))
    if do_mahalanobis:
        res = getMahalanobisDists(pde.patches, pde.get_mahalanobis_ellipsoid, delta, n_points=100, rank=rank, jacfac=jacfac, maxeigs=maxeigs)
        sio.savemat("DSqr.mat", res)
        res = sio.loadmat("DSqr.mat")
        DSqr, mask = res["gamma"], res["mask"]
    else:
        D = np.sum(pde.patches**2, 1)[:, None]
        DSqr = D + D.T - 2*pde.patches.dot(pde.patches.T)
    DSqr[DSqr < 0] = 0

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
    plt.imshow(D, aspect='auto', cmap=cmap)
    plt.show()
    if make_video:
        Y = Y[:, 0:3]
        pde.makeVideo(Y, D, skip=1, cmap=cmap)
    
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
    ks = KSSimulation(scale=0.5)
    testKS_Diffusion(ks, pd = (32, 32), nsamples=(40, 40), \
        dMaxSqr=10, delta=3, rank=2, maxeigs=40, jacfac=1,\
        rotate=False, use_rotinvariant=False, do_mahalanobis=True)
    #testKS_Variations(ks)
    #testKS_Rotations(ks)
