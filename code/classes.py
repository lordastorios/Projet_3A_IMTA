# -*- coding: utf-8 -*-

"""classes.py: This file defines the classes used for eddies detection."""

__author__ = "G. Ghienne, A. Lefebvre, A. Lerosey, L. Menard, A. Perier"
__date__ = "December 2020"
__version__ = "1.0"
__maintainer__ = "Not maintened"
__email__ = [
    "guillaume.ghienne@imt-atlantique.net",
    "alexandre.lefebvre@imt-atlantique.net",
    "antoine.lerosey@imt-atlantique.net",
    "luc.menard@imt-atlantique.net",
    "alexandre.perier@imt-atlantique.net",
]

import numpy as np

from AnDA_codes.AnDA_analog_forecasting import AnDA_analog_forecasting

from tools import grad_desc, estimate_sea_level_center
from constants import EQUATORIAL_EARTH_RADIUS
import netCDF4 as nc
import scipy.interpolate as sp_interp


class StreamLine:
    """Class used to represent a stream line and compute its caracteristics.

    The stream line is represented by a serie of points, using spherical
    coordinates (longitude, latitude). All the caracteristics of this stream
    line are computed from this list.

    Args:
        coord_list (array_like) : List of coordinates representing the stream
            line.
        delta_time (array_like or float) : Delta time between 2 consecutive
            points. It should be a float if the delta time is a constant or an
            array_like with len = n-2 and n beeing the number of points
            otherwise (n-2 because it is used for computing the  angular
            velocity).
        cut_lines (bool, default = True) : If True, the considered streamline
            will be the shortest sub streamline so that the winding angle is
            equal to 2 pi. This parameter has not effect if the winding angle of
            the complet streamline is lower than 2 pi.

    Attributes:
        coord_list (array_like) : List of coordinates representing the stream
            line.
        nb_points (int) : Number of points in coord_list.
        length (float) : Total length of the line.
        mean_pos (ndarray(2,)) : Average of the coordinates. If the stream line
            is in a eddy, it represents an approximation of the eddy center.
        winding_angle (float) : Sum of the oriented angles formed by 3
            consecutive points in the line. Unit is the radian
        angular_velocities (np.array) : List of the oriented angular velocities
            between 3 consecutive points in the line. Unit is the rad.s**-1

    """

    def __init__(self, coord_list, delta_time, cut_lines=True):
        self.coord_list = np.array(coord_list, dtype=float)
        self.nb_points = len(self.coord_list)
        self.mean_pos = np.mean(coord_list, axis=0)
        self._set_length()
        self._set_winding_angle_and_angular_velocity(delta_time, cut_lines)

    def _set_length(self):
        """ Compute the total length of the streamline. """
        if self.nb_points <= 1:
            self.length = 0
        else:
            ldiff_degree = self.coord_list[1:] - self.coord_list[:-1]
            ldiff_meter = ldiff_degree * np.pi * EQUATORIAL_EARTH_RADIUS / 180
            ldiff_meter[:, 0] *= np.cos(self.mean_pos[1] * np.pi / 180)
            self.length = np.sum(
                np.sqrt(ldiff_meter[:, 0] ** 2 + ldiff_meter[:, 1] ** 2)
            )

    def _set_winding_angle_and_angular_velocity(self, delta_time, cut_lines):
        """Compute the sum of the oriented angles in the line.

        The angle at a given point X_k is the angle between vect(X_k-1, X_k) and
        vect(X_k, X_k+1). If the winding angle is higher than 2 pi, the
        streamline is cut to keep the shortest streamline so that the winding
        angle is greater or equal to 2 pi.

        Args
            delta_time (array_like or float) : Delta time between 2 consecutive
                points. If dt is a constant delta_time should be a float and an
                array_like with len = n-1 otherwise, n beeing the number of
                points.
        cut_lines (bool) : If True, the considered streamline will be the
            shortest sub streamline so that the winding angle is equal to 2 pi.
            This parameter has not effect if the winding angle of the complet
            streamline is lower than 2 pi.

        """

        coord = self.coord_list[:, 0] + 1j * self.coord_list[:, 1]
        coord += coord.real * (np.cos(self.mean_pos[1] * np.pi / 180) - 1)
        vectors = coord[1:] - coord[:-1]

        # Remove the null vectors
        not_null_vect = vectors != 0
        vectors = vectors[not_null_vect]
        norms = abs(vectors)
        angles = np.angle(vectors[1:]/vectors[:-1])

        if type(delta_time * 1.0) != float:
            delta_time = delta_time[not_null_vect]

        self.coord_list = self.coord_list[np.append(not_null_vect, True), :]
        self.nb_points = len(self.coord_list)

        if self.nb_points <= 2:
            self.winding_angle = 0
            self.angular_velocities = 0
        else:
            # Find the shortest sub streamline with winding angle >= 2 pi (only
            # if it exist).
            if cut_lines:
                cumsum_norms = np.cumsum(norms)

                min_ratio = 1

                min_end_id = 0
                min_start_id = self.nb_points - 2

                for start_id in range(self.nb_points - 1):
                    start_norm = cumsum_norms[start_id]
                    start_pts = coord[start_id]
                    for end_id in range(start_id, self.nb_points - 1):
                        curr_len = cumsum_norms[end_id] - start_norm
                        start_end_dist = abs(coord[end_id] - start_pts)
                        ratio = start_end_dist / curr_len
                        if ratio <= min_ratio:
                            min_ratio = ratio
                            min_end_id = end_id
                            min_start_id = start_id

                # Recompute the first attributs after taking the sub streamline
                self.coord_list = np.array(
                    self.coord_list[min_start_id : min_end_id + 1], dtype=float
                )
                self.nb_points = (min_end_id - min_start_id) + 1
                self.length = cumsum_norms[end_id] - cumsum_norms[start_id]
                self.mean_pos = np.mean(self.coord_list, axis=0)

                # Recompute the sub angle list and delta_time
                angles = np.array(angles[min_start_id : min_end_id - 1])
                if type(delta_time * 1.0) != float:
                    delta_time = np.array(delta_time[min_start_id:min_end_id])

            # Set the winding angle and the angular velocity
            self.winding_angle = np.sum(angles)

            if type(delta_time * 1.0) == float:
                self.angular_velocities = angles / delta_time
            else:
                self.angular_velocities = (
                    2 * angles / (delta_time[1:] + delta_time[:-1])
                )

    def get_mean_radius(self):
        """Compute the mean radius of the stream line.

        Compute the mean distance between the stream line mean pos and the list
        of coordinates. This value does not have any real meaning if the stream
        line winding is low.

        Returns:
            mean_radius (float) : The mean radius of the stream line.

        """

        radius = np.array(self.coord_list)
        radius[:, 0] -= self.mean_pos[0]
        radius[:, 1] -= self.mean_pos[1]
        radius = np.sqrt(np.sum(radius ** 2, axis=1))
        mean_radius = np.mean(radius)
        return mean_radius


class Eddy:
    """Class used to represent a eddy and compute its caracteristics.

    Args:
        sl_list (list of StreamLine) : List of streamline representing the eddy.
        param (list, default=[]) : Parameters of the eddy when it is not
            initialized from a list of streamlines. It should contain 6 values.
            The 2 first are the longitude and the latitude of the eddy center.
            The 2 next are the axis length. The 5th is the angle between the
            first axis of the ellipse and the x-axis (longitudes). The last is
            the angular speed. The parameter 'sl_list' is ignored if this
            parameter is not the default one.

    Attributes:
        sl_list (list of StreamLine) : List of stream line representing the
            eddy.
        date : date at which the eddy is detected
        nb_sl (int) : Number of stream lines in sl_list.
        center (array_like(2)) : Mean of stream lines center weigthed by the
            stream line length. This represent the stream line center.
        cov_matrix (ndarray(2,2)) : Covariance of all the points on all the
            streamlines.
        axis_len (ndarray(2) : Length of the eddy axis.
        axis_dir (ndarray(2,2)) : Direction of the eddy axis, they are computed
            using the covariance of all the points on all the streamlines.
        angular_velocity (float) : Mean angular velocity of the eddy.

    """

    def __init__(self, sl_list, date, param=[]):
        if len(param) == 0:
            self.date = date
            self.sl_list = list(sl_list)
            self.nb_sl = len(self.sl_list)
            self._set_center()
            self._set_axis_len_and_dir()
            self._set_angular_velocity()
        elif len(param) == 6:
            self.date = date
            self.sl_list = []
            self.nb_sl = 0
            self.center = np.array(param[:2])
            self.axis_len = np.array(param[2:4])
            self.axis_dir = np.array(
                [
                    [np.cos(param[4]), -np.sin(param[4])],
                    [np.sin(param[4]), np.cos(param[4])],
                ]
            )
            self.angular_velocity = param[5]
        else:
            print("Parameter(s) missing for initialising an eddy")

    def _set_center(self):
        """Compute the eddy center.

        The center is the mean of stream lines center weigthed by the stream
        line length.

        """
        sl_center = np.array([self.sl_list[k].mean_pos for k in range(self.nb_sl)])
        sl_nb_pts = np.array([self.sl_list[k].nb_points for k in range(self.nb_sl)])
        sl_wcenter = [sl_center[k] * sl_nb_pts[k] for k in range(self.nb_sl)]
        self.center = np.sum(sl_wcenter, axis=0) / np.sum(sl_nb_pts)

    def _set_axis_len_and_dir(self):
        """Compute the axis direction and lengths for the ellipse

        The eigen vectors are the directions of the axis of the ellipse. The
        length is computed so that sum{(x,y)}{(x/a)**2+(y/b)**2-1} is minimum.

        """

        # Compute the covariance matrixe for finding the axis direction
        nb_coord = np.sum([self.sl_list[k].nb_points for k in range(self.nb_sl)])
        merged_coord_list = np.zeros((nb_coord, 2))
        index = 0
        for sl_id in range(self.nb_sl):
            sl = self.sl_list[sl_id]
            merged_coord_list[index : index + sl.nb_points, :] = np.array(sl.coord_list)
            index += sl.nb_points
        merged_coord_list[:, 0] -= self.center[0]
        merged_coord_list[:, 1] -= self.center[1]
        cov_matrix = np.cov(merged_coord_list.T)
        self.axis_dir = np.linalg.eig(cov_matrix)[1]

        # Rotate and center the points so that the ellipse equation can be
        # written "(x/a)**2 + (y/b)**2 = 1"
        angles = np.angle(self.axis_dir[:, 0] + 1j * self.axis_dir[:, 1])
        angle = angles[0]
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        aligned_points = np.dot(rotation, merged_coord_list.T).T

        # Compute the regression square error for given (a,b) parameters
        def error(a, b):
            points_sq = aligned_points ** 2
            points_sq[:, 0] /= max(1.0 * a * a, 0.0001)
            points_sq[:, 1] /= max(1.0 * b * b, 0.0001)
            return np.sum((np.sqrt(np.sum(points_sq, axis=1)) - 1) ** 2)

        # Gradient of the square error
        def grad_error(a, b):
            points_sq = aligned_points ** 2
            x2 = np.array(points_sq[:, 0])
            y2 = np.array(points_sq[:, 1])
            sq_coeff = np.sqrt(x2 / (a * a) + y2 / (b * b))
            common_coeff = -2 * (1 - 1 / sq_coeff)
            grad_a = np.sum(common_coeff * x2) / (a ** 3)
            grad_b = np.sum(common_coeff * y2) / (b ** 3)
            return grad_a, grad_b

        # Gradient decente
        a, b = grad_desc(error, grad_error)
        self.axis_len = np.array([a, b])

        h0 = estimate_sea_level_center(
            self.date,
            self.center,
            aligned_points,
            self.axis_len,
            angle,
            stream_data_fname="../data/data.nc",
            R=EQUATORIAL_EARTH_RADIUS,
        )
        print("h0: ", h0)

    def _set_angular_velocity(self):
        """Compute the angular velocity of the eddy.

        The angular velocity is the angle variation per time units.

        """
        nb_angular_velocities = 0
        sum_angular_velocities = 0
        for sl_id in range(self.nb_sl):
            w_list = self.sl_list[sl_id].angular_velocities
            nb_angular_velocities += len(w_list)
            sum_angular_velocities += np.sum(w_list)
        self.angular_velocity = sum_angular_velocities / nb_angular_velocities

class Catalog:
    """Class used to represent a catalog of eddies and their successors.

    Args:
        analogs (array_like) : 6 parameters (ax 1) of different eddies (ax 0)
        successors (aaray like) : parameters of corresponding successors.

    Attributes:
        analogs (array_like) : 6 parameters (ax 1) of different eddies (ax 0)
        successors (array like) : parameters of corresponding successors.

    """

    def __init__(self, analogs, successors):
        self.analogs = analogs
        self.successors = successors

    def adjust(self,multi):
        self.analogs=self.analogs*multi
        self.successors=self.successors*multi

class Observation:
    """Class used to represent observation of eddies and give time scale.

    Args:
        values (array_like) : 6 parameters (axis 1) of different observed eddies
            (axis 0). np.nan if no observed parameter.
        time (array like) : times corresponding to observation and desired
            prediction.

    Attributes:
        values (array_like) : 6 parameters (axis 1) of different observed eddies
            (axis 0). np.nan if no observed parameter.
        time (array like) : times corresponding to observation and desired
            prediction.

    """

    def __init__(self,values,time):
        self.values = values
        self.time = time

    def adjust(self,multi):
        self.values =self.values*multi

class ForecastingMethod:
    """Class used to set parameters for the forecasting model.

    Args:
        k (int) : number of analogs to use during the forecast.
        neighborhood (array like(6,6)) : a_ij = 1 or 0. 1 if the ith parameter
            will be used to predict the jth parameter, 0 otherwise
        catalog(Catalog) : a catalog of analogs and successors
        regression(String) : chosen regression ('locally_constant', 'increment',
            'local_linear')
    Attributes:
        k (int) : number of analogs to use during the forecast.
        neighborhood (array like(6,6)) : a_ij = 1 or 0. 1 if the ith parameter
            will be used to predict the jth parameter, 0 otherwise
        catalog(Catalog) : a catalog of analogs and successors
        regression(String) : chosen regression ('locally_constant', 'increment',
            'local_linear')
        sampling(String) : chosen sampler ('gaussian')

    """

    def __init__(self,catalog,k=50,regression="increment"):
        self.k=min(k,np.shape(catalog.analogs)[0]-1)
        self.neighborhood=np.ones((6,6))
        self.catalog=catalog
        self.regression=regression
        self.sampling="gaussian"

class FilteringMethod:
    """Class used to set parameters for the filtering model.

    Args:
        method (String) : chosen method ('AnEnKF', 'AnEnKS')
        N (int) : number of members
        B (array_like(6,6)) : modele noise covariance matrix
        R (array_like(6,6)) : observation noise covariance matrix
        forecasting_model (ForecastingModel) : ForecastingModel instance
    Attributes:
        method (String) : chosen method ('AnEnKF', 'AnEnKS')
        N (int) : number of members
        H (array_like(6,6)) : matrix of state transition
        B (array_like(6,6)) : modele noise covariance matrix
        R (array_like(6,6)) : observation noise covariance matrix
        forecasting_model (ForecastingModel) : ForecastingModel instance
        xb (array_like(1,6)) : parameters of the initial eddy

    """
    def __init__(self,B,R,forecasting_method,method="AnEnKS",N=50):
        self.method=method
        self.N=N
        self.xb=None
        self.H=np.eye(6)
        self.B=B
        self.R=R
        self.AF=forecasting_method

    def set_first_eddy(self,xb):
        self.xb=xb

    def adjust_R(self,multi):
        self.R=self.R*multi

    def adjust_B(self,multi):
        self.B=self.B*multi

    def m(self,x):
        return AnDA_analog_forecasting(x,self.AF)
