import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from scipy import interpolate
from scipy import stats
import scipy.fftpack as sfft
from scipy import ndimage
from Kuramoto import *

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

def crosscorr2d_angle(p1, p2, theta):
    """
    Perform an area-normalized cross-correlation
    after undoing a rotation of the second patch
    """
    p1pad = im2fft2_padded(p1, window=False, shift=False)
    r1pad = im2fft2_padded(np.ones_like(p1), window=False, shift=False)
    p2pad = im2fft2_padded(p2, window=False, shift=False, flip=False, theta=theta)
    r2pad = im2fft2_padded(np.ones_like(p2), window=False, flip=False, shift=False, theta=theta)
    pscorr = np.abs(sfft.ifft2(p1pad['fppad']*p2pad['fppad']))
    rscorr = np.abs(sfft.ifft2(r1pad['fppad']*r2pad['fppad']))
    #rscorr = correlate2d(r1pad['ppad'], r2pad['ppad'], 'same')
    rscorr[rscorr < 1] = 1.0
    corr = pscorr/rscorr
    return {'corr':pscorr, 'p1pad':p1pad, 'p2pad':p2pad}


if __name__ == '__main__':
    dim = 64
    fac = 0.5
    ks = KSSimulation(co_rotating=False, scale=(fac*7, fac/2))
    ks.I = ks.I[0:int(195*fac), :]
    #ks.I = np.random.randn(ks.I.shape[0], ks.I.shape[1])
    vmax = np.max(np.abs(ks.I))
    vmin = -vmax
    
    x1 = np.array([120, 100])
    x2 = x1 + np.array([10, 10])
    
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
        ks.thetas = np.array([0, theta])
        ks.completeObservations()
        p1 = np.reshape(ks.patches[0, :], (dim, dim))
        p2 = np.reshape(ks.patches[1, :], (dim, dim))
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
        res2 = crosscorr2d_angle(p1, p2, theta_est+np.pi)
        if np.max(res2['corr']) < np.max(res['corr']):
            res = res2
            theta_est = (theta_est+np.pi) % (2*np.pi)
        corr = res['corr']
        idxcorr = np.argmax(corr)
        idxcorr = np.unravel_index(idxcorr, corr.shape)

        #"""
        plt.clf()
        plt.subplot(231)
        plt.imshow(res['p1pad']['ppad'])
        plt.subplot(232)
        plt.imshow(res['p2pad']['ppad'])
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
        plt.title("theta=%.3g, theta_est=%.3g"%(theta, theta_est))
        plt.savefig("%i.png"%i)
        #"""

        thetas.append(theta)
        thetas_est.append(theta_est)
    
    thetas = np.array(thetas)
    thetas_est = np.array(thetas_est)
    slope, intercept, r_value, p_value, std_err = stats.linregress(thetas, thetas_est)

    plt.figure()
    plt.scatter(thetas, thetas_est)
    plt.legend(["y = %.3g x + %.3g"%(slope, intercept)])
    plt.axis('equal')
    plt.xlabel("Thetas")
    plt.ylabel("Thetas Estimated")
    plt.savefig("Thetas_Est.svg", bbox_inches='tight')