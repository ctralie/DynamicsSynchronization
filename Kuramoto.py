import numpy as np
import scipy.io as sio
import scipy.linalg as slinalg
from skimage.transform import resize
from scipy.ndimage.filters import gaussian_filter1d as gf1d
import matplotlib.pyplot as plt
from PDE2D import *
from Mahalanobis import *

class KSSimulation(PDE2D):
    """
    Attributes
    ----------
    (In addition to attributes from PDE2D)

    xcoords: ndarray(NObservations)
        Locations of patch centers in radians
    tcoords: ndarray(NObservations)
        Locations of patch centers in seconds
    """
    def __init__(self, co_rotating = False, scale=(1.0, 1.0), reldir='./'):
        PDE2D.__init__(self)
        res = sio.loadmat("%s/KS.mat"%reldir)
        self.I = res["data"]
        self.c = res["c"] # Traveling wave factor
        self.scale = scale
        M, N = self.I.shape[0], self.I.shape[1]
        if co_rotating:
            self.make_corotating()
        self.I = resize(self.I, (int(M*scale[0]), int(N*scale[1])), anti_aliasing=True, mode='reflect')
        self.ts = np.linspace(res["tmin"].flatten(), res["tmax"].flatten(), self.I.shape[0])
        self.xs = np.linspace(0, 2*np.pi, self.I.shape[1]+1)[0:self.I.shape[1]]
    
    def crop(self, t0, t1, x0, x1):
        self.I = self.I[t0:t1, x0:x1]
        self.ts = self.ts[t0:t1]
        self.xs = self.xs[x0:x1]
    
    def make_corotating(self, do_plot=False):
        """
        Warp into a co-rotating frame by circularly shifting each time slice
        by an appropriate amount
        """
        ratio = self.c
        INew = np.array(self.I)
        N = self.I.shape[1]
        for i in range(1, self.I.shape[0]):
            pix = np.arange(N)
            pix = np.concatenate((pix-N, pix, pix+N))
            x = self.I[i, :]
            x = np.concatenate((x, x, x))
            spl = InterpolatedUnivariateSpline(pix, x)
            INew[i, :] = spl((np.arange(N) + i*ratio) % N)
        self.I = INew

    def get_t_periodcoords(self, ts):
        """
        Assuming the period of the traveling wave is consistent,
        return an index of each row of I into the period in [0, 1]
        Parameters
        ----------
        ts: ndarray(N)
            A list of t locations
        """
        return (ts-self.ts[0])/(0.056/2) % 1
    
    def get_x_periodcoords(self, xs):
        """
        Return an index of each spatial location into [0, 1]
        Parameters
        ---------
        xs: ndarray(N)
            A list of x locations
        """
        return np.mod(xs, self.I.shape[1])/float(self.I.shape[1])

    def drawSolutionImage(self, time_extent = True, color_coords = True, inc = 0.02):
        if time_extent:
            from matplotlib.ticker import FormatStrFormatter
            m = np.max(np.abs(self.I))
            I = np.array(self.I)
            I = np.concatenate((I, I, I), 1)
            plt.imshow(I, extent = (-self.I.shape[1], 2*self.I.shape[1], self.I.shape[0], 0), cmap='RdGy', vmin=-m, vmax=m)
            d = self.I.shape[1]/2
            if color_coords:
                ys = self.get_t_periodcoords(self.ts)
                plt.scatter(-0.7*d*np.ones_like(ys)+2, np.arange(I.shape[0]), c=ys, cmap='viridis', vmin=0, vmax=1)
                xs = np.arange(-2*d, 4*d)
                plt.scatter(xs, I.shape[0]*np.ones_like(xs), c=self.get_x_periodcoords(xs), cmap='viridis', vmin=0, vmax=1)
            plt.xticks([-2*d, -d, 0, d, 2*d, 3*d, 4*d], ['$-2\\pi$', '$-\\pi$', '0', '$\\pi$', '$2 \\pi$', '$3 \\pi$', '$4 \\pi$'])
            plt.xlim([-0.7*d, d*2.7])
            ts = self.ts
            ms = inc/(ts[1]-ts[0])
            n = np.ceil(ts.size/ms)
            idx = np.arange(n)*ms
            ts = ['%.3g'%t for t in ts[np.array(idx, dtype=int)]]
            plt.yticks(idx, ts)
            plt.ylim([I.shape[0]+2, 0])
            plt.xlabel("Space (Radians)")
            plt.ylabel("Time (Seconds)")
        else:
            plt.imshow(self.I, interpolation='nearest', aspect='auto', cmap='RdGy')
    
    def concatenateOther(self, other):
        PDE2D.concatenateOther(self, other)
        self.xcoords = np.concatenate((self.xcoords, other.xcoords))
        self.tcoords = np.concatenate((self.tcoords, other.tcoords))

    def completeObservations(self):
        """
        Save the plot coordinates for x and t as a side effect of the ordinary
        completion of the observations
        """
        self.xcoords = self.get_x_periodcoords(self.Xs)
        tcoords = np.array(np.round(self.Ts), dtype=int)
        self.tcoords = self.get_t_periodcoords(self.ts[tcoords])
        PDE2D.completeObservations(self)

    def resort_byidx(self, idx):
        PDE2D.resort_byidx(self, idx)
        self.xcoords = self.xcoords[idx]
        self.tcoords = self.tcoords[idx]

    def makeVideo(self, Y, D = np.array([]), skip=20, cmap='viridis'):
        """
        Use the previously computed plot coordinates to color the result
        """
        PDE2D.makeVideo(self, Y=Y, D=D, skip=skip, cmap=cmap, colorvar1=self.xcoords, colorvar2=self.tcoords)

    def makeTimeSeriesVideo(self):
        plt.figure(figsize=(12, 4))
        m = np.max(np.abs(self.I))
        for i, idx in enumerate(range(0, self.I.shape[0], 10)):
            plt.clf()
            plt.subplot(121)
            self.drawSolutionImage(True)
            plt.colorbar()
            plt.plot([0, 2*np.pi], [self.ts[idx]]*2)
            plt.xlabel("Space (Radians)")
            plt.ylabel("Time (Seconds)")
            plt.title("Spacetime Grid")
            plt.subplot(122)
            plt.plot(self.xs, self.I[idx, :])
            plt.ylim([-m*1.1, m*1.1])
            plt.xlabel("Space (Radians)")
            plt.title("Timeslice at %g Seconds"%self.ts[idx])
            plt.savefig("%i.png"%i, bbox_inches='tight')