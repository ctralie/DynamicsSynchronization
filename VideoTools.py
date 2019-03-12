"""
Programmer: Chris Tralie
Purpose: Some tools that load/save videos in Python
"""
import numpy as np
import numpy.linalg as linalg
import time
import os
import subprocess
import matplotlib.image as mpimage
import scipy.misc
import scipy.signal
import sys
from PIL import Image

#Need these for saving 3D video

AVCONV_BIN = 'ffmpeg'
TEMP_STR = "pymeshtempprefix"

#############################################################
####                  VIDEO I/O TOOLS                   #####
#############################################################

#Methods for converting to YCbCr (copied matrices from Matlab)
toNTSC = np.array([[0.2989, 0.5959, 0.2115], [0.587, -0.2744, -0.5229], [0.114, -0.3216, 0.3114]])
fromNTSC = np.linalg.inv(toNTSC)

def rgb2ntsc(F):
    return F.dot(toNTSC.T)

def ntsc2rgb(F):
    return F.dot(fromNTSC.T)

def rgb2gray(F, repDims = True):
    G = np.dot(F[...,:3], [0.299, 0.587, 0.114])
    if repDims:
        ret = np.zeros((G.shape[0], G.shape[1], 3))
        for k in range(3):
            ret[:, :, k] = G
        return ret
    else:
        return G

def cleanupTempFiles():
    files = os.listdir('.')
    for f in files:
        if f.find(TEMP_STR) > -1:
            os.remove(f)

def loadImageIOVideo(path):
    if not os.path.exists(path):
        print("ERROR: Video path not found: %s"%path)
        return None
    import imageio
    videoReader = imageio.get_reader(path, 'ffmpeg')
    NFrames = videoReader.get_length()
    I = None
    for i in range(0, NFrames):
        frame = videoReader.get_data(i)    
        if I is None:
            I = np.zeros((NFrames, frame.size))
            IDims = frame.shape
        I[i, :] = np.array(frame.flatten(), dtype = np.float32)/255.0
    return (I, IDims)

def saveFrames(I, IDims, frame_dir='frames/'):
    for idx in range(I.shape[0]):
        frame = np.reshape(I[idx,:],IDims)
        rescaled_frame = (255.0*frame).astype(np.uint8)
        Image.fromarray(rescaled_frame).save(frame_dir+'frame-'+str(idx)+'.jpg')

#Output video
#I: PxN video array, IDims: Dimensions of each frame
def saveVideo(I, IDims, filename, FrameRate = 30, YCbCr = False, Normalize = False):
    #Overwrite by default
    if os.path.exists(filename):
        os.remove(filename)
    N = I.shape[0]
    if YCbCr:
        for i in range(N):
            frame = np.reshape(I[i, :], IDims)
            I[i, :] = ntsc2rgb(frame).flatten()
    if Normalize:
        I = I-np.min(I)
        I = I/np.max(I)
    for i in range(N):
        frame = np.reshape(I[i, :], IDims)
        frame[frame < 0] = 0
        frame[frame > 1] = 1
        mpimage.imsave("%s%i.png"%(TEMP_STR, i+1), frame)
    if os.path.exists(filename):
        os.remove(filename)
    #Convert to video using avconv
    command = [AVCONV_BIN,
                '-r', "%i"%FrameRate,
                '-i', TEMP_STR + '%d.png',
                '-r', "%i"%FrameRate,
                '-b', '30000k',
                filename]
    subprocess.call(command)
    #Clean up
    for i in range(N):
        os.remove("%s%i.png"%(TEMP_STR, i+1))


#############################################################
####        SLIDING WINDOW VIDEO TOOLS, GENERAL         #####
#############################################################
def getPCAVideo(I):
    ICov = I.dot(I.T)
    [lam, V] = linalg.eigh(ICov)
    lam[lam < 0] = 0
    V = V*np.sqrt(lam[None, :])
    return V

def getSlidingWindowVideo(I, dim, Tau, dT):
    N = I.shape[0] #Number of frames
    P = I.shape[1] #Number of pixels (possibly after PCA)
    pix = np.arange(P)
    NWindows = int(np.floor((N-dim*Tau)/dT))
    X = np.zeros((NWindows, dim*P))
    idx = np.arange(N)
    for i in range(NWindows):
        idxx = dT*i + Tau*np.arange(dim)
        start = int(np.floor(idxx[0]))
        end = int(np.ceil(idxx[-1]))
        f = scipy.interpolate.interp2d(pix, idx[start:end+1], I[idx[start:end+1], :], kind='linear')
        X[i, :] = f(pix, idxx).flatten()
    return X

def getSlidingWindowVideoInteger(I, dim):
    N = I.shape[0]
    M = N-dim+1
    X = np.zeros((M, I.shape[1]*dim))
    for i in range(X.shape[0]):
        X[i, :] = I[i:i+dim, :].flatten()
    return X

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    frames = []
    for i, idx in enumerate(range(2200, 3700)):
        IM = mpimage.imread("bacteria/%i.png"%idx)
        IM = rgb2gray(IM)[:, :, 0]
        plt.imshow(IM)
        plt.savefig("%i.png"%i)