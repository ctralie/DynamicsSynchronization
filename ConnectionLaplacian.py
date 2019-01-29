"""
Programmer: Chris Tralie
Purpose: An implementation of the Connection Laplacian
for point clouds in R^d
"""
import numpy as np
import numpy.linalg as linalg
from scipy import sparse 
from scipy.sparse.linalg import lsqr, cg, eigsh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from LocalPCA import *


"""#####################################################
                VECTOR DIFFUSION MAPS
#####################################################"""


def getConnectionLaplacian(ws, Os, N, k, weighted=True):
    """
    Given a set of weights and corresponding orientation matrices,
    return k eigenvectors of the connection Laplacian.
    Parameters
    ----------
    ws: ndarray(M, 3)
        An array of weights for each included edge
    Os: M-element list of (ndarray(d, d))
        The corresponding orthogonal transformation matrices
    N: int
        Number of vertices in the graph
    k: int
        Number of eigenvectors to compute
    weighted: boolean
        Whether to normalize by the degree
    
    Returns
    -------
    w: ndarray(k)
        Array of k eigenvalues
    v: ndarray(N*d, k)
        Array of the corresponding eigenvectors
    """
    d = Os[0].shape[0]
    W = sparse.coo_matrix((ws[:, 2], (ws[:, 0], ws[:, 1])), shape=(N, N)).tocsr()
    ## Step 1: Create D^-1 matrix
    deg = np.array(W.sum(1)).flatten()
    deg[deg == 0] = 1

    ## Step 2: Create S matrix
    I = []
    J = []
    V = []
    Jsoff, Isoff = np.meshgrid(np.arange(d), np.arange(d))
    Jsoff = Jsoff.flatten()
    Isoff = Isoff.flatten()
    for idx in range(ws.shape[0]):
        [i, j, wij] = ws[idx, :]
        i, j = int(i), int(j)
        wijOij = wij*Os[idx]
        if weighted:
            wijOij /= deg[i]
        wijOij = (wijOij.flatten()).tolist()
        I.append((i*d + Isoff).tolist())
        J.append((j*d + Jsoff).tolist())
        V.append(wijOij)
    I, J, V = np.array(I).flatten(), np.array(J).flatten(), np.array(V).flatten()
    S = sparse.coo_matrix((V, (I, J)), shape=(N*d, N*d)).tocsr()
    w, v = eigsh(S, which='LA', k=k)
    # Put largest first
    idx = np.argsort(-w)
    w = w[idx]
    v = v[:, idx]
    return w, v


def getConnectionLaplacianPC(X, k, gammadim, eps_pca, eps_w, \
                            K_usqr_pca = lambda u: np.exp(-5*u)*(u <= 1), \
                            K_usqr_w = lambda u: np.exp(-5*u)*(u <= 1)):
    
    """
    Compute eigenvectors of a connection Laplacian estimated from Local
    PCA on a point cloud
    Parameters
    ----------
    X: ndarray(N, p)
        A point cloud with N points in p dimensions
    k: int
        Number of eigenvectors to compute
    gammadim: float
        Explained variance ratio for local PCA
    eps_pca: float
        The epsilon to use when doing local PCA
    eps_w: float
        The epsilon to use when computing the affinity matrix
    K_usqr_pca: function float->float
        Kernel function for local PCA
    K_usqr_w: function float->float
        Kernel function for affinity matrix
    
    Returns
    -------
    w: ndarray(k)
        Array of k eigenvalues
    v: ndarray(N*d, k)
        Array of the corresponding eigenvectors
    bases: N-length list of ndarray(p, d)
        All of the orthonormal basis matrices for each point
    """
    N = X.shape[0]

    ## Step 1: Perform local PCA
    bases = getLocalPCA(X, eps_pca, gammadim, K_usqr_pca)

    ## Step 2: Compute the affinity weights and the
    # orthogonal transformation matrices between points
    # which are connected to each other
    ws = []
    Os = []
    for i in range(N):
        dsqr = getdists_sqr(X, i, exclude_self=True)
        di = K_usqr_w(dsqr/eps_w)
        OiT = bases[i].T
        for w, j in zip(di[di > 0], np.arange(N)[di > 0]):
            ws.append([i, j, w])
            U, _, VT = linalg.svd(OiT.dot(bases[j]))
            Oij = U.dot(VT)
            Os.append(Oij)
    ws = np.array(ws)
    
    ## Step 3: Compute the connection Laplacian
    w, v = getConnectionLaplacian(ws, Os, N, k, weighted=True)
    return w, v, bases



"""#####################################################
                        EXAMPLES
#####################################################"""

def testConnectionLaplacianSquareGrid(N, seed=0, torus = False):
    """
    Randomly rotate square tiles on an NxN grid
    Add increasing amounts of noise to some of the Os
    Parameters
    ----------
    N: int
        Dimension of grid
    seed: int
        Seed for random initialization of vector angles
    torus: boolean
        Whether this grid is thought of as on the torus or not
    """
    np.random.seed(seed)
    thetas = 2*np.pi*np.random.rand(N, N)
    ws = []
    Os = []
    for i in range(N):
        for j in range(N):
            idxi = i*N+j
            for [di, dj] in [[-1, 0], [1, 0], [0, 1], [0, -1]]:
                i2 = i+di
                j2 = j+dj
                if torus:
                    i2, j2 = i2%N, j2%N
                else:
                    if i2 < 0 or j2 < 0 or i2 >= N or j2 >= N:
                        continue
                idxj = i2*N+j2
                ws.append([idxi, idxj, 1.0])
                # Oij moves vectors from j to i
                theta = thetas[i2, j2] - thetas[i, j]
                c = np.cos(theta)
                s = np.sin(theta)
                Oij = np.array([[c, -s], [s, c]])
                Os.append(Oij)
    ws = np.array(ws)
    w, v = getConnectionLaplacian(ws, Os, N**2, 2)
    print(w)
    ax = plt.gca()
    for idx in range(N*N):
        i, j = np.unravel_index(idx, (N, N))
        plt.scatter([j], [i], 20, 'k')
        for k in [0]:
            vidx = v[idx*2:(idx+1)*2, k]
            # Bring into world coordinates
            c = np.cos(thetas[i, j])
            s = np.sin(thetas[i, j])
            Oij = np.array([[c, -s], [s, c]])
            vidx = Oij.dot(vidx)
            vidx = 0.5*vidx/np.sqrt(np.sum(vidx**2))
            #vidx = Oij[:, 0]*0.5
            ax.arrow(j, i, vidx[1], vidx[0], head_width=0.1, head_length=0.2)
    plt.axis('equal')
    plt.show()



def getTorusKnot(p, q, pt):
    """
    Return a p-q torus knot parameterized on [0, 1]
    Parameters
    ----------
    p: int
        p parameter for torus knot
    q: int
        q parameter for torus knot
    pt: ndarray(N)
        Parameterization on the interval [0, 1]
    """
    N = len(pt)
    t = 2*np.pi*pt
    X = np.zeros((N, 3))
    r = np.cos(q*t) + 2
    X[:, 0] = r*np.cos(p*t)
    X[:, 1] = r*np.sin(p*t)
    X[:, 2] = -np.sin(q*t)
    return X

def getNDEllipse(u, v, pt):
    """
    Get an N-dimensional ellipse point cloud
    Parameters
    ----------
    u: ndarray(1, N)
        First axis
    v: ndarray(1, N)
        Second axis
    pt: ndarray(M)
        Parameterization in the interval [0, 1]
    """
    t = 2*np.pi*pt
    t = t[:, None]
    return np.cos(t)*u + np.sin(t)*v


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def testConnectionLaplacian3D():
    np.random.seed(10)
    X = getTorusKnot(2, 3, np.random.rand(1000))
    #X = getNDEllipse(np.random.randn(1, 3), np.random.randn(1, 3), np.random.rand(100))
    #X += 0.05*np.random.randn(X.shape[0], X.shape[1])
    eps = 1.0
    _, v, bases = getConnectionLaplacianPC(X, k=2, gammadim=0.9, eps_pca = eps, eps_w = eps)
    d = bases[0].shape[1]

    (perm, _) = getGreedyPerm(X)
    perm = perm[0:50]
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(121, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2])
    for idx in perm:
        x0 = X[idx, :]
        u = bases[idx][:, 0]
        x = np.concatenate((x0[None, :], x0[None, :] + 0.3*u[None, :]), 0)
        a = Arrow3D(x[:, 0], x[:, 1], x[:, 2], mutation_scale=20, \
                lw=2, arrowstyle="-|>", color="r")
        ax.add_artist(a)

    ax = plt.subplot(122, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2])
    for idx in perm:
        x1 = X[idx, :]
        Oi = bases[idx]
        ui = v[idx*d:(idx+1)*d, 1]
        ui = Oi.dot(ui) # Transform into world coordinates
        x2 = x1 + 0.5*ui
        x = np.concatenate((x1[None, :], x2[None, :]), 0)
        a = Arrow3D(x[:, 0], x[:, 1], x[:, 2], mutation_scale=20, \
                lw=2, arrowstyle="-|>", color="r")
        ax.add_artist(a)

    plt.show()


def testConnectionLaplacian2D():
    np.random.seed(10)
    ts = np.random.rand(100)
    X = getNDEllipse(np.random.randn(1, 2), np.random.randn(1, 2), ts)
    eps = 0.5
    _, v, bases = getConnectionLaplacianPC(X, k=2, gammadim=0.8, eps_pca = eps, eps_w = eps)
    d = bases[0].shape[1]

    (perm, _) = getGreedyPerm(X)
    perm = perm[0:20]
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.scatter(X[:, 0], X[:, 1])
    ax = plt.gca()
    for idx in perm:
        x0 = X[idx, :]
        u = bases[idx][:, 0]
        u = 0.1*u/np.sqrt(np.sum(u**2))
        ax.arrow(x0[0], x0[1], u[0], u[1], head_width=0.1, head_length=0.2)
    plt.axis('equal')
    plt.title("Local PCA")

    plt.subplot(122)
    plt.scatter(X[:, 0], X[:, 1])
    ax = plt.gca()
    for idx in perm:
        x0 = X[idx, :]
        Oi = bases[idx]
        u = v[idx*d:(idx+1)*d, 1]
        u = Oi.dot(u) # Transform into world coordinates
        u = 0.1*u/np.sqrt(np.sum(u**2))
        ax.arrow(x0[0], x0[1], u[0], u[1], head_width=0.1, head_length=0.2)
    plt.axis('equal')
    plt.title("Local PCA + Connection Laplacian")

    plt.show()

if __name__ == '__main__':
    #testConnectionLaplacian2D()
    #testConnectionLaplacian3D()
    testConnectionLaplacianSquareGrid(10)