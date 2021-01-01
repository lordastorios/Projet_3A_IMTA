# -*- coding: utf-8 -*-

"""classes.py: This file defines the functions used for ploting eddies and
streamlines."""

__author__     = "G. Ghienne, A. Lefebvre, A. Lerosey, L. Menard, A. Perier"
__date__       = "December 2020"
__license__    = "Libre"
__version__    = "1.0"
__maintainer__ = "Not maintened"
__email__      = ["guillaume.ghienne@imt-atlantique.net",
                  "alexandre.lefebvre@imt-atlantique.net",
                  "antoine.lerosey@imt-atlantique.net",
                  "luc.menard@imt-atlantique.net",
                  "alexandre.perier@imt-atlantique.net"]

import cartopy
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse

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
        self.fig,self.ax=plt.subplots(1,1,subplot_kw={'projection':projection})

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
            self.ax.plot(traj[:,0], traj[:,1], line_style,
                         transform=cartopy.crs.Geodetic())

    def plot_eddies(self, eddies, plot_traj=True, line_style='-'):
        """Plots eddies from a .nc particles trajectories file.

        Plot the center of the eddies and an ellipse representing each eddy.

        Args:
            eddies (list of Eddy or Eddy) : Eddy or list of Eddy to show.
            plot_traj (bool, default=True) : If True, plot the trajectories.
            line_style (str) : line style for the trajectories representation.
        """

        if type(eddies) == Eddy:
            eddies = [eddies]

        n = len(eddies)

        # Plot centers
        centers  = np.array([eddies[k].center for k in range(n)])
        if len(eddies)>0:
            self.ax.plot(centers[:,0],centers[:,1],'k+',
                         transform=cartopy.crs.Geodetic())

        # Plot trajectories
        if plot_traj:
            for k in range(n):
                self.plot_trajectories(eddies[k].sl_list,line_style=line_style)

        # Plot ellipses
        # The angle of the ellipse is computed with 'vect_x - vect_y*1j' because of y-axis inversion for plotting ellispses.
        axes_len = np.array([eddies[k].axis_len for k in range(n)])
        axes_dir = np.array([eddies[k].axis_dir for k in range(n)])
        angles =   np.array([np.angle(axes_dir[k,0,0]-axes_dir[k,0,1]*1j) for k
                             in range(n)])/(2*np.pi)*360
        for k in range(n):
            ellipse = Ellipse(centers[k,:],axes_len[k,0]*3,axes_len[k,1]*3,
                              angle=angles[k],color='black',alpha=1,fill=False,
                              linewidth=2)
            self.ax.add_patch(ellipse)

    def plot(self,X,Y,line_style='+'):
        self.ax.plot(X,Y,line_style,transform=cartopy.crs.Geodetic())

    def show(self):
        plt.show()
