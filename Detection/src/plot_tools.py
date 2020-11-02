import cartopy
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from classes import StreamLine, Eddy


class StreamPlot:
    """ Matplotlib figure with specific methods for ploting trajectories, vortexes etc.

    Matplotlib figure with specific methods for ploting trajectories, vortexes,
    ellipses, from .nc files or from numpy arrays

    Attributes:
        fig (matplotlib.figure.Figure) : Matplotlib figure
        ax (cartopy.mpl.geoaxes.GeoAxesSubplot) : cartopy projection

    """

    def __init__(self):
        projection = cartopy.crs.PlateCarree()
        self.fig, self.ax = plt.subplots(1, 1, subplot_kw={'projection': projection})

        gl = self.ax.gridlines(crs=projection, draw_labels=True)
        gl.xlabels_top, gl.ylabels_right = (False, False)
        gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
        gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER

        self.ax.coastlines()


    def plot_trajectories(self, trajectories, line_style='-'):
        """Plots trajectories on a map.

        Input parameter can be a .nc file, a StreamLine object or a list of
        StreamLine object.

        Args:
            trajectories (str or list of StreamLine or StreamLine) : If it is a
                string, name of the .nc file containing trajectories data. It
                should have been created using parcels.ParticleSet.ParticleFile.
                Otherwise StreamLine or list of StreamLine to show.
            line_style (str) : line style for the trajectories representation.
        """

        if type(trajectories) == str:
            try:
                pfile = xr.open_dataset(str(trajectories), decode_cf=True)
            except:
                pfile = xr.open_dataset(str(trajectories), decode_cf=False)
            lon = np.ma.filled(pfile.variables['lon'], np.nan)
            lat = np.ma.filled(pfile.variables['lat'], np.nan)
            pfile.close()

            for p in range(lon.shape[1]):
                lon[:, p] = [ln if ln < 180 else ln - 360 for ln in lon[:, p]]

            self.ax.plot(lon.T, lat.T, '-', transform=cartopy.crs.Geodetic())
            return

        if type(trajectories) == StreamLine:
            trajectories = [trajectories]

        nb_traj = len(trajectories)
        for k in range(nb_traj):
            traj = trajectories[k].coord_list
            self.ax.plot(traj[:,0], traj[:,1], line_style, transform=cartopy.crs.Geodetic())

    def plot_eddies(self, eddies, line_style='+'):
        """Plots eddies from a .nc particles trajectories file.

        Args:
            eddies (list of Eddy or Eddy) : Eddy or list of Eddy to show.
            line_style (str) : line style for the trajectories representation.
        """

        if type(eddies) == Eddy:
            eddies = [eddies]

        centers = np.array([eddies[k].center for k in range(len(eddies))])
        if len(eddies)>0:
            self.ax.plot(centers[:,0],centers[:,1], line_style,
                         transform=cartopy.crs.Geodetic())

    def plot(self,X,Y,line_style='+'):
        self.ax.plot(X,Y,line_style,transform=cartopy.crs.Geodetic())

    def show(self):
        plt.show()
