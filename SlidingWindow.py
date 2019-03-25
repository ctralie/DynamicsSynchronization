import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

def drawLineColored(idx, x, C, ax = None):
    """
    Since matplotlib doesn't allow plotting lines with
    colors to my knowledge, I made a function to do that
    Parameters
    ----------
    idx: ndarray(M)
        Indices into the time series to plot
    x: ndarray(N)
        Original time series
    C: ndarray(M, 3)
        An array of colors
    ax: matplotlib.axes 
        Optional axis object to which to plot
    """
    if not ax:
        ax = plt.gca()
    for i in range(len(idx)-1):
        ax.plot(idx[i:i+2], x[i:i+2], c=C[i, :])

def getSlidingWindow(x, dim, Tau, dT):
    """
    Return a sliding window of a time series,
    using arbitrary sampling.  Use linear interpolation
    to fill in values in windows not on the original grid
    Parameters
    ----------
    x: ndarray(N)
        The original time series
    dim: int
        Dimension of sliding window (number of lags+1)
    Tau: float
        Length between lags, in units of time series
    dT: float
        Length between windows, in units of time series
    Returns
    -------
    X: ndarray(N, dim)
        All sliding windows stacked up
    """
    N = len(x)
    NWindows = int(np.floor((N-dim*Tau)/dT))
    X = np.zeros((NWindows, dim))
    spl = InterpolatedUnivariateSpline(np.arange(N), x)
    for i in range(NWindows):
        idxx = dT*i + Tau*np.arange(dim)
        start = int(np.floor(idxx[0]))
        end = int(np.ceil(idxx[-1]))+2
        # Only take windows that are within range
        if end >= len(x):
            X = X[0:i, :]
            break
        X[i, :] = spl(idxx)
    return X

def getSlidingWindowNoInterp(x, dim):
    """
    Return a sliding window of a time series, with all
    samples on the original time series grid (Tau=1, dT=1)
    Parameters
    ----------
    x: ndarray(N)
        The original time series
    dim: int
        Dimension of sliding window (number of lags+1)
    Returns
    -------
    X: ndarray(N, dim)
        All sliding windows stacked up
    """
    N = len(x)
    NWindows = N - dim + 1
    X = np.zeros((NWindows, dim))
    idx = np.arange(N)
    for i in range(NWindows):
        X[i, :] = x[i:i+dim]
    return X


def get_lims(X, dim, pad=0.1):
    """
    Return the limits around a dimension with some padding
    Parameters
    ----------
    X: ndarray(N, d)
        Point cloud in d dimensions
    dim: int
        Dimension to extract limits from
    pad: float
        Factor by which to pad
    """
    xlims = [np.min(X[:, dim]), np.max(X[:, dim])]
    xlims[0] = xlims[0]-(xlims[1]-xlims[0])*pad
    xlims[1] = xlims[1]+(xlims[1]-xlims[0])*pad
    return xlims

#A class for doing animation of the sliding window
class SlidingWindowAnimator(animation.FuncAnimation):
    """
    Create a video of a sliding window time series, plotting the windows
    evolving on the left, and a dimension reduced version of the time series
    on the right
    """
    def __init__(self, filename, fig, x, Y, dim, Tau, dT, hop=1, fps=30, bitrate=10000):
        """
        Parameters
        ----------
        filename: string
            Output name of video
        fig: matplotlib figure handle
            Handle of figure where this will be drawn
        x: ndarray(N)
            Original time series
        Y: ndarray(M, dim_proj)
            A dimension reduced version of the sliding window to draw
        dim: int
            Dimension of sliding window
        Tau: float
            Lag interval between samples in sliding window
        dT: float
            Sample interval between sliding windows
        hop: int
            Hop between windows in adjacent frames of video (good to make a higher
            number for time series with a lot of sliding windows)
        fps: int
            Frames per second of the video
        bitrate: int
            Output bitrate of the video
        """
        assert(Y.shape[1] >= 2)
        self.fig = fig
        self.x = x
        self.Y = Y
        self.dim = dim
        self.Tau = Tau
        self.dT = dT
        Win = dim*Tau
        self.Win = Win
        self.bgcolor = (0.15, 0.15, 0.15)
        self.hop = hop
        t = dT*np.arange(Y.shape[0]) # Start times of each window
        
        c = plt.get_cmap('Spectral')
        C = c(np.array(np.round(255*t/np.max(t)), dtype=np.int32))
        tx = np.arange(len(x))
        tx[tx > np.max(t)] = np.max(t)
        Cx = c(np.array(np.round(255*tx/np.max(t)), dtype=np.int32))
        Cx = Cx[:, 0:3]
        self.C = C[:, 0:3]
        self.xlims = get_lims(x[:, None], 0, 0.2)


        ax1 = fig.add_subplot(121)
        ax1.set_facecolor(self.bgcolor)
        if Y.shape[1] >= 3:
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.set_xlim(get_lims(Y, 0))
            ax2.set_ylim(get_lims(Y, 1))
            ax2.set_zlim(get_lims(Y, 2))
        else:
            ax2 = fig.add_subplot(122)
            ax2.set_xlim(get_lims(Y, 0))
            ax2.set_ylim(get_lims(Y, 1))
            ax2.set_facecolor(self.bgcolor)
        self.ax1 = ax1
        self.ax2 = ax2

        #Original time series
        drawLineColored(np.arange(x.size), x, Cx, ax1)
        c = C[0, :]
        self.windowPlot, = ax1.plot([0], [x[0]], c=c[None, :])
        self.leftLim, = ax1.plot([0, 0], self.xlims, c=c[None, :])
        self.rightLim, = ax1.plot([Win, Win], self.xlims, c=c[None, :], lineWidth=2)

        #Setup animation thread
        self.n_frames = int(np.floor(Y.shape[0]/float(hop)))
        animation.FuncAnimation.__init__(self, fig, func = self._draw_frame, frames = self.n_frames, interval = 10)

        #Write movie
        FFMpegWriter = animation.writers['ffmpeg']
        metadata = dict(title='Sliding Window Animation',
                        comment='Awesome vids by Chris Tralie! xD')
        writer = FFMpegWriter(fps=fps, metadata=metadata, bitrate = bitrate)
        self.save(filename, writer = writer)

    def _draw_frame(self, i):
        print("Rendering frame %i of %i"%(i+1, self.n_frames))
        idxs = np.arange(i*self.hop, (i+1)*self.hop)
        i1 = self.dT*idxs[-1]
        i2 = i1 + self.Win
        i1 = int(np.round(i1))
        i2 = int(np.round(i2))
        c = self.C[idxs[-1], :]
        self.windowPlot.set_xdata(np.arange(i1, i2))
        self.windowPlot.set_ydata(self.x[i1:i2])
        self.windowPlot.set_color(c)
        self.leftLim.set_xdata([i1, i1])
        self.leftLim.set_ydata(self.xlims)
        self.leftLim.set_color(c)
        self.rightLim.set_xdata([i2, i2])
        self.rightLim.set_ydata(self.xlims)
        self.rightLim.set_color(c)
        Y = self.Y
        if Y.shape[1] == 2:
            self.ax2.scatter(Y[idxs, 0], Y[idxs, 1], c=c[None, :])
        elif Y.shape[1] >= 3:
            self.ax2.scatter(Y[idxs, 0], Y[idxs, 1], Y[idxs, 2], c=c[None, :])


def doSinesExample():
    from sklearn.decomposition import PCA
    T = 20 #The period in number of samples
    NPeriods = 15 #How many periods to go through
    N = T*NPeriods #The total number of samples
    t = np.linspace(0, 2*np.pi*NPeriods, N+1)[0:N] #Sampling indices in time
    x = np.cos(t) + np.cos((1+np.sqrt(5))*t/2) #The final signal
    
    dim = 40
    Tau = 0.5
    dT = np.pi/6
    X = getSlidingWindow(x, dim, Tau, dT)
    pca = PCA(n_components = 3)
    Y = pca.fit_transform(X)
    eigs = pca.explained_variance_
    
    fig = plt.figure(figsize=(12, 6))
    a = SlidingWindowAnimator("out.mp4", fig, x, Y, dim, Tau, dT, hop=5)


if __name__ == '__main__':
    doSinesExample()