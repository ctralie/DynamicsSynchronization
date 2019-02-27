import numpy as np
import scipy.io as sio
import scipy.linalg as slinalg
import matplotlib.pyplot as plt
from PDE2D import *
from PatchDescriptors import *

class KSSimulation(PDE2D):
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



def testKS_Mahalanobis(pd, nsamples, delta, rotate=False):
    f_patch = lambda x: x
    if rotate:
        f_patch = lambda patches: get_derivative_shells(patches, pd, orders=[0, 1], n_shells=50)
    ks = KSSimulation()
    ks.makeObservations(pd, nsamples, buff=delta, rotate=rotate, f_patch=f_patch)
    DSqr = ks.getMahalanobisDists(delta=delta, n_points=100, d=5)
    Y = doDiffusionMaps(DSqr, ks.Xs, dMaxSqrCoeff = 1.0)
    plt.show()
    
    perm, lambdas = getGreedyPerm(Y, 600)
    plt.figure()
    dgms = ripser(Y[perm, :], maxdim=2)["dgms"]
    plot_dgms(dgms, show=False)
    plt.show()
    
    ks.makeVideo(Y, skip=15)


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
    ks.makeObservations((80, 40), (20, 20), rotate=True)
    
    ks.plotPatches(False)
    plt.show()
    ks.plotPatches()

if __name__ == '__main__':
    #testKS_NLDM(pd = (64, 64), nsamples=(94, 201), dMaxSqrCoeff=10, skip=1)
    testKS_Mahalanobis(pd = (50, 50), nsamples=(94, 150), delta=1, rotate=True)
    #testKS_Variations()
    #testKS_Rotations()
