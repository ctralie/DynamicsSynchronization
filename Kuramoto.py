import numpy as np
import scipy.io as sio
import scipy.linalg as slinalg
from skimage.transform import resize
import matplotlib.pyplot as plt
from PDE2D import *
from PatchDescriptors import *
from Mahalanobis import *

class KSSimulation(PDE2D):
    def __init__(self, scale=(1.0, 1.0)):
        PDE2D.__init__(self)
        res = sio.loadmat("KS.mat")
        I = res["data"]
        I = resize(I, (I.shape[0]*scale[0], I.shape[1]*scale[1]), anti_aliasing=True, mode='reflect')
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



def testKS_Diffusion(pde, pd, nsamples, delta, rotate=False, use_rotinvariant = False, dMaxSqr=1.0, do_mahalanobis=False, rank=2, jacfac=1, maxeigs=10, noisefac=0.0, do_tda=False, make_video=True, cmap='magma_r'):
    Y = testMahalanobis_PDE2D(pde, pd=pd, nsamples=nsamples, \
                    dMaxSqr=dMaxSqr, delta=3, rank=2, maxeigs=maxeigs, jacfac=1,\
                    periodic=True, rotate=False, do_mahalanobis=True, \
                    precomputed_samples=(400, 400), pca_dim=10, do_plot=True)
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
    ks = KSSimulation(scale=(0.5, 0.25))
    testKS_Diffusion(ks, pd = (32, 32), nsamples=10000,
        dMaxSqr=10, delta=5, rank=2, maxeigs=5, jacfac=1, noisefac=0,
        rotate=False, use_rotinvariant=False, do_mahalanobis=True)
    #testKS_Variations(ks)
    #testKS_Rotations(ks)
