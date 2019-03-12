import numpy as np
import scipy.io as sio
import scipy.linalg as slinalg
import matplotlib.pyplot as plt
from PDE3D import *
from PatchDescriptors import *
from DiffusionMaps import *
import matplotlib.image as mpimage

class GL2DSimulation(PDE3D):
    def __init__(self):
        res = sio.loadmat("cgle_2d.mat")
        I = np.real(res["data"])
        self.I = I

    def saveFrames(self):
        vmin = np.min(self.I)
        v = (self.I-vmin)
        v /= np.max(v)
        c = plt.get_cmap('RdGy')
        C = c(np.array(np.round(255.0*v.flatten()), dtype=np.int32))
        C = C[:, 0:3]
        C = np.reshape(C, (v.shape[0], v.shape[1], v.shape[2], 3))
        for i in range(C.shape[0]):
            mpimage.imsave("%i.png"%i, C[i, :, :, :])
    
    def get_timeseries(self):
        I = np.moveaxis(self.I, 0, -1)
        ii, jj = np.meshgrid(np.arange(I.shape[0]), np.arange(I.shape[1]), indexing='ij')
        X = np.reshape(I, (I.shape[0]*I.shape[1], I.shape[2]))
        D = np.sum(X**2, 1)[:, None]
        DSqr = D + D.T - 2*X.dot(X.T)
        Y = doDiffusionMaps(DSqr, X[:, 0], dMaxSqrCoeff=0.6, do_plot=False)
        plt.scatter(Y[:, 0], Y[:, 1], c=X[:, 0])
        plt.show()




def testFullFrameStack():
    gl = GL2DSimulation()
    gl.makeObservations(pd=(10, 120, 120), nsamples=(500, 1, 1), buff=(1, 1, 1))
    D = np.sum(gl.patches**2, 1)[:, None]
    DSqr = D + D.T - 2*gl.patches.dot(gl.patches.T)
    Y = doDiffusionMaps(DSqr, gl.Xs, dMaxSqrCoeff=4.0, do_plot=False)
    #gl.makeVideo(Y[:, 0:2], plot_boundaries=False, colorvar=gl.Ts)
    theta = np.arctan2(Y[:, 1], Y[:, 0])
    idxs = np.argsort(theta)
    c = plt.get_cmap('magma_r')
    C = c(np.array(np.round(255.0*idxs/np.max(idxs)), dtype=np.int32))
    C = C[:, 0:3]
    C = C[idxs, :]
    Y = Y[idxs, :]
    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(idxs):
        plt.clf()
        plt.subplot(121)
        t = int(np.round(gl.Ts[idx]))
        gl.draw_frame(t)
        plt.title("Frame %i"%t)
        plt.subplot(122)
        plt.scatter(Y[:, 0], Y[:, 1], 100, c=np.array([[0, 0, 0, 0]]))
        plt.scatter(Y[0:i+1, 0], Y[0:i+1, 1], c=C[0:i+1, :])
        plt.savefig("%i.png"%i)

def testRecoverSquareTimeSeries():
    gl = GL2DSimulation()
    
    D = np.sum(gl.patches**2, 1)[:, None]
    DSqr = D + D.T - 2*gl.patches.dot(gl.patches.T)
    Y = doDiffusionMaps(DSqr, gl.Xs, dMaxSqrCoeff=10, do_plot=False)
    plt.scatter(Y[:, 0], Y[:, 1], c=gl.patches[:, 0])
    plt.show()


if __name__ == '__main__':
    #testFullFrameStack()
    #testRecoverSquareTimeSeries()
    gl = GL2DSimulation()
    gl.get_timeseries()