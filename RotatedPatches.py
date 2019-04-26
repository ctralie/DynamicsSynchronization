"""
Fast FFT-based techniques to uncover rotations between 
partially overlapping, rotated patches
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.fftpack as sfft
from scipy import ndimage
from scipy.signal import fftconvolve
import time

def pad_patch(p, window=True):
    """
    Zeropad a patch so that it can be shifted fully left/right, up/down
    Parameters
    ----------
    p: ndarray(d1, d2)
        Input patch
    window: boolean
        Whether to apply a hanning window to the patch
    Returns
    -------
    ppad: ndarray(n, n), n > d1, d2
        The zeropadded patch
    """
    pd = p.shape
    if window:
        hy = np.hanning(pd[0])
        hx = np.hanning(pd[1])
        p = p*hy[:, None]
        p = p*hx[None, :]
    r_max = int(np.ceil(np.sqrt((pd[0]/2)**2 + (pd[1]/2)**2)))
    # Figure out padding for patch
    dy = int(np.ceil((r_max-pd[0]/2.0)))
    dx = int(np.ceil((r_max-pd[1]/2.0)))
    ppad = np.zeros((2*r_max+pd[0]+2*dy, 2*r_max+pd[1]+2*dx))
    ppad[dy+r_max:pd[0]+dy+r_max, dx+r_max:pd[1]+dx+r_max] = p
    return ppad

def im2polarfft(p, r_max = None, nr = None, ntheta = None, window = True):
    """
    Compute a polar coordinate representation of a 2D FFT obtained
    after padding a patch
    Parameters
    ----------
    p: ndarray(d1, d2)
        The patch
    r_max: int
        Maximum radius to consider in polar coordinates
    nr: int
        Number of subdivisions of radius from 0 to r_max
    ntheta: int
        Number of angles to take in [0, 2*pi).  If None, this is chosen as the minimum
        number that avoids aliasing
    
    Returns
    -------
    dtheta: float
        The angle increment, in radians, between adjacent columns in the polar representation
    polar: ndarray(nr, ntheta)
        The polar representation of the zeropadded patch
    """
    ppad = pad_patch(p, window)
    fppad = sfft.fftshift(sfft.fft2(ppad))
    pd = fppad.shape
    if not r_max:
        r_max = np.sqrt((pd[0]/2)**2 + (pd[1]/2)**2)
    pixx = np.arange(pd[1]) - float(pd[1])/2
    pixy = np.arange(pd[0]) - float(pd[0])/2
    # Make an adequate resolution in theta to avoid aliasing
    if not ntheta:
        ntheta = 2*int(2**np.ceil(np.log(2*np.pi*r_max)/np.log(2)))
    thetas = np.linspace(0, 2*np.pi, ntheta+1)[0:ntheta]
    if not nr:
        nr = int(np.round(2*r_max))
    rs = np.linspace(0, r_max, nr)
    xs = rs[:, None]*np.cos(thetas[None, :])
    ys = rs[:, None]*np.sin(thetas[None, :])
    xs, ys = xs.flatten(), ys.flatten()
    f = interpolate.RectBivariateSpline(pixy, pixx, np.abs(fppad))
    return thetas[1]-thetas[0], np.reshape(f(xs, ys, grid=False), (rs.size, thetas.size))    

def crosscorr2d_angle(p1, p2, ptheta):
    """
    Perform cross-correlation after undoing a rotation of the second patch
    Try theta and theta+pi
    Parameters
    ----------
    p1: ndarray(m1, n1)
        Patch 1
    p2: ndarray(m2, n2)
        Patch 2
    ptheta: float
        Initial estimated angle by which to rotate p2, in radians
    Returns
    -------
    {'corr': ndarray(M, N)
        The correlation image,
     'theta': float
        }
    """
    max_theta = ptheta
    max_corr_val = 0
    max_corr_img = np.array([])
    corr_vals = []
    for dtheta in [0, np.pi]:
        theta = ptheta + dtheta
        p2rot = ndimage.rotate(p2, angle=theta*180/np.pi, reshape=True)
        corr_img = fftconvolve(p1, np.fliplr(np.flipud(p2rot)))
        corr_val = np.max(corr_img)
        corr_vals.append(theta)
        corr_vals.append(corr_val)
        if corr_val > max_corr_val:
            max_corr_val = corr_val
            max_corr_img = corr_img
            max_theta = theta
    return {'corr':max_corr_img, 'theta':max_theta, 'corr_vals':corr_vals}


def get1dfftpolar(p, r_max, nr, ntheta, flip=False):
    """
    Helper function for computing angle-wise FFT of each row
    a patch's 2D FFT in polar coordinates
    """
    dtheta, polar = im2polarfft(p, r_max, nr, ntheta)
    if flip:
        polar = np.fliplr(polar)
    fft = np.fft.fft(polar, axis=1)
    return dtheta, fft, polar

def estimate_rotangle(p1, p2, r_max=20, nr=40, ntheta=None, fft1=np.array([]), fft2=np.array([])):
    """
    Parameters
    ----------
    p1: ndarray(d, d)
        First patch
    p2: ndarray(d, d)
        Second patch
    r_max: int
        Maximum radial frequency to consider in polar coordinates
    nr: int
        Number of subdivisions of radius from 0 to r_max
    ntheta: int
        Number of angles to take in [0, 2*pi).  If None, this is chosen as the minimum
        number that avoids aliasing
    fft1: ndarray(N)
        Precomputed fft1, if it exists
    fft2: ndarray(N)
        Precomputed fft2, if it exists
    
    Returns
    -------
    A bunch of stuff in a dictionary.  The most important stuff is
    {'theta_est': float
        Estimated optimal CCW rotation to take p1 to p2, 
     'corr': ndarray(M, M)
        Array of estimated translational correlations to take p1
        to p2 after the optimal rotation (max location is not usually
        reliable in these applications because there is a lot of 
        vertical symmetry,
      'x':ndarray(n_theta)
        Scores at different theta shifts)
     }
    """
    
    if not fft1:
        dtheta, fft1, polar1 = get1dfftpolar(p1, r_max, nr, ntheta)
    else:
        polar1 = np.array([])
    dtheta, fft2, polar2 = get1dfftpolar(p2, r_max, nr, ntheta, flip=True)

    # Perform 1D circular convolution
    x = np.fft.ifft(fft1*fft2, axis=1)
    x = np.sum(x, 0)
    idx = np.argmax(x)
    theta_est = idx*dtheta % (2*np.pi)

    # Try theta and theta + pi to see which gives a better
    # normalized translational cross-correlation
    res = crosscorr2d_angle(p1, p2, theta_est)
    theta_est = res['theta']
    corr = res['corr']
    corr_vals = tuple(res['corr_vals'])

    return {'fft1':fft1, 'polar1':polar1, 'fft2':fft2, 'polar2':polar2, 'theta_est':theta_est, 'corr':corr, 'corr_vals':corr_vals, 'x':x, 'dtheta':dtheta}


def testRotationRecovery(save_frames = True):
    from Kuramoto import KSSimulation
    np.random.seed(0)
    dim = 64
    fac = 0.5
    ks = KSSimulation(co_rotating=False, scale=(fac*7, fac/2))
    ks.I = ks.I[0:int(195*fac), :]
    #ks.I = np.random.randn(ks.I.shape[0], ks.I.shape[1])
    vmax = np.max(np.abs(ks.I))
    vmin = -vmax

    x1 = np.array([120, 100])
    x2 = x1 + np.array([20, 20])
    
    ks.Xs = np.array([x1[0], x2[0]])
    ks.Ts = np.array([x1[1], x2[1]])
    ks.pd = (dim, dim)

    plt.figure(figsize=(15, 10))
    thetas = []
    thetas_est = []
    for i in range(100):
        ks.thetas = 2*np.pi*np.random.rand(2)
        ks.completeObservations()
        p1 = np.reshape(ks.patches[0, :], (dim, dim))
        p2 = np.reshape(ks.patches[1, :], (dim, dim))

        res = estimate_rotangle(p1, p2)
        theta_est, corr_vals, corr = res['theta_est'], res['corr_vals'], res['corr']
        polar1, polar2, x = res['polar1'], res['polar2'], res['x']
        idx = np.argmax(x)
        thetas.append((ks.thetas[1]-ks.thetas[0]) % (2*np.pi))
        thetas_est.append(theta_est)
        idxcorr = np.unravel_index(np.argmax(corr), corr.shape)

        if save_frames:
            plt.clf()
            plt.subplot(231)
            plt.imshow(p1, vmin=vmin, vmax=vmax)
            plt.title("%.3g: %.3g,\n%.3g: %.3g"%corr_vals)
            plt.subplot(232)
            plt.imshow(p2, vmin=vmin, vmax=vmax)
            plt.subplot(233)
            plt.imshow(corr)
            plt.colorbar()
            plt.scatter([idxcorr[1]], [idxcorr[0]])
            plt.title("(%i, %i)"%tuple(idxcorr))
            plt.subplot(234)
            plt.imshow(polar1, aspect='auto')
            plt.subplot(235)
            plt.imshow(polar2, aspect='auto')
            plt.subplot(236)
            plt.plot(x)
            plt.stem([idx], x[idx:idx+1])
            plt.title("theta=%.3g, theta_est=%.3g"%(thetas[-1], thetas_est[-1]))
            plt.savefig("%i.png"%i)


    thetas = np.array(thetas)
    thetas_est = np.array(thetas_est)
    diffs = np.abs(thetas-thetas_est)
    diffs = np.minimum(diffs, np.abs(thetas-thetas_est+2*np.pi))
    diffs = np.minimum(diffs, np.abs(thetas-thetas_est-2*np.pi))
    

    plt.figure()
    plt.scatter(thetas, thetas_est)
    plt.axis('equal')
    plt.xlabel("Thetas")
    plt.ylabel("Thetas Estimated")
    #plt.title("Avg Diff: %.3g, Stand Diff: %.3g"%(np.mean(diffs), np.std(diffs)))
    plt.savefig("Thetas_Est.svg", bbox_inches='tight')

if __name__ == '__main__':
    testRotationRecovery()