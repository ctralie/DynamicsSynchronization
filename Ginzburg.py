import numpy as np
import scipy.io as sio
import scipy.linalg as slinalg
import matplotlib.pyplot as plt
from PDE2D import *
from PatchDescriptors import *

class GLSimulation(PDE2D):
    def __init__(self):
        res = sio.loadmat("cgle_1d.mat")
        I = np.abs(res["data"])
        self.I = I

def testVerticalRemix():
    gl = GLSimulation()
    gl.makeObservationsIntegerGrid(pd=(gl.I.shape[0], 1), sub=(1, 1))

    D = np.sum(gl.patches**2, 1)[:, None]
    DSqr = D + D.T - 2*gl.patches.dot(gl.patches.T)
    print(np.max(DSqr))
    Y = doDiffusionMaps(DSqr, gl.Xs, dMaxSqrCoeff=4.0, do_plot=False)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(gl.I, aspect='auto', cmap='jet')
    plt.subplot(122)
    plt.plot(Y[:, 0])
    plt.show()

if __name__ == '__main__':
    testVerticalRemix()