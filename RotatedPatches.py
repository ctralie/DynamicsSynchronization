import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.fftpack as sfft
from scipy import ndimage
from scipy.signal import fftconvolve
from Kuramoto import *
import time

def im2fft2_padded(p, theta=None, window=True, flip = False, shift=True):
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
    if theta:
        ppad = ndimage.rotate(ppad, angle=theta*180/np.pi, reshape=False)
    if flip:
        ppad = np.fliplr(np.flipud(ppad))
    fppad = sfft.fft2(ppad)
    if shift:
        fppad = sfft.fftshift(fppad)
    return {'ppad':ppad, 'fppad':fppad}

def im2polar(p, r_max = None, nr = None, ntheta = None):
    pd = p.shape
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
    f = interpolate.RectBivariateSpline(pixy, pixx, p)
    return thetas[1]-thetas[0], np.reshape(f(xs, ys, grid=False), (rs.size, thetas.size))    

def crosscorr2d_angle(p1, p2, ptheta):
    """
    Perform cross-correlation after undoing a rotation of the second patch
    Try (+/- theta) + [0, pi]
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


if __name__ == '__main__':
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
    r_max = 20
    nr = 40
    ntheta = None

    plt.figure(figsize=(15, 10))
    thetas = []
    thetas_est = []
    for i, theta in enumerate(np.linspace(0, 2*np.pi, 100)):
        ks.thetas = 2*np.pi*np.random.rand(2)
        ks.completeObservations()
        p1 = np.reshape(ks.patches[0, :], (dim, dim))
        p2 = np.reshape(ks.patches[1, :], (dim, dim))

        tic = time.time()
        ppad1 = np.abs(im2fft2_padded(p1)['fppad'])
        ppad2 = np.abs(im2fft2_padded(p2)['fppad'])
        dtheta, polar1 = im2polar(ppad1, r_max, nr, ntheta)
        dtheta, polar2 = im2polar(ppad2, r_max, nr, ntheta)

        # Perform 1D circular convolution
        fft1 = np.fft.fft(polar1, axis=1)
        fft2 = np.fft.fft(np.fliplr(polar2), axis=1)
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
        idxcorr = np.argmax(corr)
        idxcorr = np.unravel_index(idxcorr, corr.shape)

        print("Elapsed Time: %.3g"%(time.time()-tic))
        thetas.append((ks.thetas[1]-ks.thetas[0]) % (2*np.pi))
        thetas_est.append(theta_est)

        """
        plt.clf()
        plt.subplot(231)
        plt.imshow(p1, vmin=vmin, vmax=vmax)
        plt.title("%.3g: %.3g, %.3g: %.3g\n%.3g: %.3g, %.3g: %.3g"%corr_vals)
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
        """

        
    
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