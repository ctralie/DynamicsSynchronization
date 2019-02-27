import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import torch
from kymatio import Scattering2D
from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter1d as gf1d
from sklearn.decomposition import PCA

def get_spinimage(im, n_angles=50, do_plot=False):
    """
    Make a spin image
    Parameters
    ----------
    im: ndarray(N, M):
        Image to rotate
    n_angles: int
        Number of angles by which to rotate
    """
    M, N = im.shape[0], im.shape[1]
    pixx = np.arange(N) - float(N)/2
    pixy = np.arange(M) - float(M)/2
    f = interpolate.RectBivariateSpline(pixy, pixx, im)
    D = int(np.ceil(np.sqrt(M**2+N**2)))
    
    pix = np.arange(D) - float(D)/2
    x, y = np.meshgrid(pix, pix)
    x = x.flatten()
    y = y.flatten()
    I = np.zeros((D, D))
    if do_plot:
        plt.figure(figsize=(12, 6))
    for i, t in enumerate(np.linspace(0, 2*np.pi, n_angles+1)[0:n_angles]):
        c, s = np.cos(t), np.sin(t)
        xt = c*x + s*y
        yt = -s*x + c*y
        It = f(yt, xt, grid=False)
        It[np.abs(yt) > float(M)/2] = 0
        It[np.abs(xt) > float(N)/2] = 0
        It = np.reshape(It, I.shape)
        I += It
        if do_plot:
            plt.clf()
            plt.subplot(121)
            plt.imshow(It, interpolation='none')
            plt.subplot(122)
            plt.imshow(I)
            plt.savefig("%i.png"%i)
    return I

def get_shells(patches, pd, n_shells, r_max = None):
    """
    Return integration through concentric circular shells, 
    normalized by the area of these shells
    Parameters
    ----------
    patches: ndarray(n_patches, pd[0]*pd[1])
        All of the patches
    pd: tuple(int, int)
        The height x width of each patch
    n_shells: int
        The number of concentric shells to take
    r_max: float
        The max radius up to which to go.  If None,
        then take the radius to be sqrt((pd[0]/2)^2 + (pd[1]/2)^2)
    Returns
    -------
    shells: ndarray(n_patches, n_shells)
        The result of integration in each shell for each patch
    """
    if not r_max:
        r_max = np.sqrt((pd[0]/2.0)**2 + (pd[1]/2.0)**2)
    centers = np.linspace(0, r_max, n_shells+1)[0:-1]
    sigma = centers[1]-centers[0]
    centers += sigma/2

    pixx = np.arange(pd[1]) - pd[1]/2.0
    pixy = np.arange(pd[0]) - pd[0]/2.0
    x, y = np.meshgrid(pixx, pixy)
    x = x.flatten()
    y = y.flatten()
    rs = np.sqrt(x**2 + y**2)
    shells = np.zeros((patches.shape[0], n_shells))
    for i, c in enumerate(centers):
        weights = np.exp(-(rs-c)**2/(2*sigma**2))
        shells[:, i] = np.sum(weights[None, :]*patches, 1)/np.sum(weights)
    return shells

def get_derivative_shells(patches, pd, n_shells, shells_fn = get_shells, orders = [0, 1, 2], r_max = None):
    """
    Compute shell descriptors on the raw patch and after derivative
    filters applied to the patch.
    """
    patchesim = np.reshape(patches, (patches.shape[0], pd[0], pd[1]))
    all_shells = []
    for order in orders:
        if order == 0:
            shells = shells_fn(patches, pd, n_shells, r_max)
        else:
            imx = gf1d(patchesim, sigma=1, order=order, axis=2)
            imy = gf1d(patchesim, sigma=1, order=order, axis=1)
            patchesorder = np.sqrt(imx**2 + imy**2)
            shells = shells_fn(np.reshape(patchesorder, patches.shape), pd, n_shells, r_max)
        # Normalize each by the standard deviation so they are comparable
        shells /= np.std(shells)
        all_shells.append(shells)
    return np.concatenate(tuple(all_shells), 1)


def get_pc_histograms(patches, n_bins=50, n_pcs=3):
    """
    Compute principal components of histograms
    """
    N = patches.shape[0]
    H = np.zeros((N, n_bins))
    lims = (np.min(patches), np.max(patches))
    for i in range(N):
        H[i, :] = np.histogram(patches[i, :], range=lims, bins=n_bins)[0]
    pca = PCA(n_components=n_pcs)
    return pca.fit_transform(H)


def get_scattering(patches, pd, J=4, L=8, rotinvariant=True):
    """
    you need to average all paths of the form
    (theta1, theta2=theta1+delta)
    for varying theta1 and fixed delta.

    so for example with L=4, average (0, 0) with (1, 1), (2, 2) etc.
    then average (0, 1) with (1, 2), (2, 3), etc.
    then average (0, 2) with (1, 3), (2, 0), etc.
    then average (0, 3) with (1, 0), (2, 1), etc.

    that invariance will bring the number of orientations from L*L to L
    """
    #dim = max(pd[0], pd[1])
    #dim = int(2**np.ceil(np.log(dim)/np.log(2)))
    dim = pd[0]

    print("Initializing scattering transform...")
    scattering = Scattering2D(shape=(dim, dim), J=J, L=L)
    print("Computing scattering coefficients...")
    WTemp = torch.zeros((1, patches.shape[0], pd[0], pd[1]))
    WTemp[0, :, :, :] = torch.from_numpy(np.reshape(patches, (patches.shape[0], pd[0], pd[1])))
    coeffs = scattering(WTemp).numpy()
    # 1 zero order, J first order, and (J, 2)*L second order averaged paths
    #res = np.zeros((patches.shape[0], 2+L))
    return np.reshape(coeffs, (patches.shape[0], coeffs.shape[2]*coeffs.shape[3]*coeffs.shape[4]))

if __name__ == '__main__':
    res = sio.loadmat("KS.mat")
    I = res["data"]
    im = I[0:64, 0:64]