"""
Utility functions shared by both PDE2D and PDE3D
Leave the functions that are different and overwritten
by PDE2D and PDE3D blank
"""
import numpy as np
import sklearn.feature_extraction.image as skimage
import skimage.transform
from sklearn.decomposition import PCA

imresize = lambda x, M, N: skimage.transform.resize(x, (M, N), anti_aliasing=True, mode='reflect')
def largeimg(D, mask=np.array([]), limit=1000):
    """
    Display an anti-aliased downsampled version of a (masked) image
    """
    res = max(D.shape[0], D.shape[1])
    if res > limit:
        fac = float(limit)/res
        retD = imresize(D, int(fac*D.shape[0]), int(fac*D.shape[1]))
        if mask.size > 0:
            maskD = imresize(mask, int(fac*D.shape[0]), int(fac*mask.shape[1]))
            retD[maskD < 0.01] = np.inf
        return retD
    else:
        retD = np.array(D)
        if mask.size > 0:
            retD[mask == 0] = np.inf
        return retD

def even_interval(k):
    """
    Return unit samples equally spaced around 0
    """
    if k%2 == 0:
        n = k/2
        return 0.5+np.arange(k)-n
    n = (k-1)/2
    return np.arange(k)-n

def approximate_rasterorder(X, Y, resy = 20):
    """
    Given a set of X and Y coordinates corresponding to
    a 2D point cloud, return an index list that visits the
    coordinates approximately in raster order
    Parameters
    ----------
    X: ndarray(N)
        X coordinates
    Y: ndarray(N)
        Y coordinates
    resy: int
        Y resolution of the grid to which to rasterize points
    Returns
    -------
    order: ndarray(N)
        Approximate raster order of the coordinates
    """
    y = Y-np.min(Y)
    y /= np.max(y)
    y = np.round(y*resy)
    idx = np.argsort(X)
    idx = idx[np.argsort(y[idx], kind='stable')]
    order = np.zeros(X.shape[0])
    order[idx] = np.arange(X.shape[0])
    return idx

def ball_rejection_sample(dim, n_points):
    """
    Sample points evenly in a ball via rejection sampling
    Parameters
    ----------
    dim: int
        Dimension of ball
    n_points: int
        Number of points in the ball
    Returns
    -------
    X: ndarray(n_points, dim)
        Sampled points in the ball
    """
    X = np.zeros((0, dim))
    while X.shape[0] < n_points:
        X2 = np.random.rand(n_points, dim) - 0.5
        X2 *= 2
        X2 = X2[np.sum(X2**2, 1) <= 1, :]
        X = np.concatenate((X, X2), 0)
    return X[0:n_points, :]

class PDEND(object):
    def __init__(self):
        self.I = np.array([[]])
        self.f_pointwise = lambda x: x
        self.f_patch = lambda x: x
        self.f_interp = None
        self.pca = None

    def getInterpolator(self):
        pass
    
    def completeObservations(self):
        pass
    
    def makeObservations(self):
        pass
    
    def get_mahalanobis_ellipsoid(self, idx, delta, n_points):
        pass

    def compose_with_dimreduction(self, pca = None, dim = np.inf):
        """
        Perform a linear dimension reduction of the patches, 
        and compose the patch function with that dimension reduction.
        There is the option to pass along a pca object from somewhere
        else, or to compute one from scratch
        Parameters
        ----------
        pca: sklearn.decomposition.PCA
            A PCA object to apply (None by default)
        dim: int
            Dimension to which to project, if no pca object
            is passed along
        """
        if not pca:
            assert dim <= self.patches.shape[1]
            print("Reducing patch dimensions from %i to %i"%(self.patches.shape[1], dim))
            pca = PCA(n_components=dim)
            self.patches = pca.fit_transform(self.patches)
            self.pca = pca
        else:
            self.patches = pca.transform(self.patches)
            self.pca = pca

    def sample_raw_patches(self):
        """
        Regardless of the patch feature functions or
        dimension reduction, sample the patches from
        scratch with raw pixels.  Save the previous
        patches and functions as a side effect, so that
        they can be recovered later
        """
        self.patches_before = np.array(self.patches)
        self.f_patch_before = self.f_patch
        self.f_pointwise_before = self.f_pointwise
        self.f_patch = lambda x: x
        self.f_pointwise = lambda x: x
        self.completeObservations()
    
    def recover_original_patches(self):
        """
        Go back to the originally sampled patches
        after having called sample_raw_patches()
        """
        self.patches = self.patches_before
        self.f_patch = self.f_patch_before
        self.f_pointwise = self.f_pointwise_before