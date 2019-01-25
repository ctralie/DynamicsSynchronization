import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import scipy.io as sio
from scipy import sparse
import time
from mpl_toolkits.mplot3d import Axes3D

def getSSM(X):
    """
    Compute a Euclidean self-similarity image between a set of points
    :param X: An Nxd matrix holding the d coordinates of N points
    :return: An NxN self-similarity matrix
    """
    D = np.sum(X**2, 1)[:, None]
    D = D + D.T - 2*X.dot(X.T)
    D[D < 0] = 0
    D = 0.5*(D + D.T)
    D = np.sqrt(D)
    return D

def getDiffusionMap(X, eps, neigs = 4, thresh=5e-4):
    """
    Perform diffusion maps with a unit timestep, automatically
    normalizing for nonuniform sampling
    Parameters
    ----------
    X: ndarray(N, d)
        A point cloud with N points in d dimensions
    eps: float
        Kernel scale parameter
    neigs: int
        Number of eigenvectors to compute
    thresh: float
        Threshold below which to zero out entries in
        the Markov chain approximation
    """
    tic = time.time()
    print("Building diffusion map matrix...")
    D = np.sum(X**2, 1)[:, None]
    DSqr = D + D.T - 2*X.dot(X.T)
    K = np.exp(-DSqr/(2*eps))
    P = np.sum(K, 1)
    P[P == 0] = 1
    KHat = (K/P[:, None])/P[None, :]
    dRow = np.sum(KHat, 1)
    KHat[KHat < thresh] = 0
    KHat = sparse.csc_matrix(KHat)
    M = sparse.diags(dRow).tocsc()

    print("Elapsed Time: %.3g"%(time.time()-tic))

    print("Solving eigen system...")
    tic = time.time()
    # Solve a generalized eigenvalue problem

    w, v = sparse.linalg.eigsh(KHat, k=neigs, M=M, which='LM')
    print("Elapsed Time: %.3g"%(time.time()-tic))
    return w[None, :]*v


def getPinchedCircle(N):
    t = np.linspace(0, 2*np.pi, N+1)[0:N]
    x = np.zeros((N, 2))
    x[:, 0] = (1.5 + np.cos(2*t))*np.cos(t)
    x[:, 1] = (1.5 + np.cos(2*t))*np.sin(t)
    return x

def getTorusKnot(N, p, q):
    t = np.linspace(0, 2*np.pi, N+1)[0:N]
    X = np.zeros((N, 3))
    r = np.cos(q*t) + 2
    X[:, 0] = r*np.cos(p*t)
    X[:, 1] = r*np.sin(p*t)
    X[:, 2] = -np.sin(q*t)
    return X

if __name__ == '__main__':
    zeroReturn = True
    N = 400
    X = getPinchedCircle(N)
    sio.savemat("X.mat", {"X":X})
    tic = time.time()
    SSMOrig = getSSM(X)
    toc = time.time()
    print("Elapsed time SSM: ", toc - tic)

    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.scatter(X[:, 0], X[:, 1], 40, np.arange(N), cmap = 'Spectral', edgecolor = 'none')
    plt.axis('equal')
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor((0.15, 0.15, 0.15))
    plt.title("Original Pinched Circle")
    plt.subplot(122)
    plt.imshow(SSMOrig, interpolation = 'nearest', cmap = 'afmhot')
    plt.title("Original SSM")
    plt.savefig("Diffusion0.svg", bbox_inches = 'tight')


    plt.figure(figsize=(12, 12))
    c = plt.get_cmap('Spectral')
    C = c(np.array(np.round(255.0*np.arange(X.shape[0])/X.shape[0]), dtype=np.int32))
    C = C[:, 0:3]
    for i, eps in enumerate(np.linspace(0.01, 2, 200)):
        M = getDiffusionMap(X, eps, 10)
        Y = M[:, [-2, -3, -4]]
        SSM = getSSM(M)

        plt.clf()
        plt.subplot(221)
        plt.imshow(M[:, 0:-1], aspect='auto')
        plt.subplot(222)
        plt.imshow(SSM, interpolation = 'nearest', cmap = 'afmhot')
        plt.title("Diffusion Distance")
        plt.subplot(223)
        plt.scatter(Y[:, 0], Y[:, 1], c=C)
        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor((0.15, 0.15, 0.15))
        plt.axis('equal')
        plt.title("2D Diffusion Map, $\epsilon = %g$"%eps)

        ax = plt.gcf().add_subplot(224, projection='3d')
        ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=C)
        plt.title("3D Diffusion Map, $\epsilon = %g$"%eps)
        

        plt.savefig("%i.png"%i, bbox_inches='tight')