import numpy as np
import scipy.io as sio
import scipy.linalg as slinalg
import matplotlib.pyplot as plt
from PDE3D import *
from PatchDescriptors import *
from DiffusionMaps import *
from VideoTools import *
from skimage import exposure

class Bacteria(PDE3D):
    def __init__(self):
        (I, IDims) = loadImageIOVideo("bacteria.avi")
        I = rgb2gray(I)[:, :, 0]
        self.I = np.reshape(I, (I.shape[0], IDims[0], IDims[1]))

def make_equalized_video():
    equalize = True
    for i in range(1, 3700):
        I = mpimage.imread("bacteria/%i.png"%i)
        I = rgb2gray(I)[:, :, 0]
        v = I-np.min(I)
        v /= np.max(v)
        if equalize:
            v = exposure.equalize_hist(v)
        c = plt.get_cmap('magma_r')
        C = c(np.array(np.round(255.0*v.flatten()), dtype=np.int32))
        C = C[:, 0:3]
        C = np.reshape(C, (I.shape[0], I.shape[1], 3))
        mpimage.imsave("%i.png"%i, C)

if __name__ == '__main__':
    make_equalized_video()