"""
Replicate the 1D time series reshuffling problem in the equal space paper
"""
import numpy as np
import scipy.io as sio
import scipy.linalg as slinalg
import matplotlib.pyplot as plt
from PDE2D import *
from PatchDescriptors import *
from DiffusionMaps import *

class GLSimulation(PDE2D):
    def __init__(self):
        PDE2D.__init__(self)
        res = sio.loadmat("cgle_1d.mat")
        I = np.abs(res["data"])
        self.I = I

def testVerticalRemix():
    gl = GLSimulation()
    gl.makeObservations(pd=(gl.I.shape[0], 1), nsamples=(1, gl.I.shape[1]))

    D = np.sum(gl.patches**2, 1)[:, None]
    DSqr = D + D.T - 2*gl.patches.dot(gl.patches.T)
    eps = np.max(DSqr)*4e-3
    Y = getDiffusionMap(DSqr, eps, distance_matrix=True, neigs=3, thresh=0)

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(gl.I, aspect='auto', cmap='jet')
    plt.subplot(122)
    plt.scatter(gl.Xs, Y[:, 0])
    plt.show()

if __name__ == '__main__':
    testVerticalRemix()