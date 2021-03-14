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
from sklearn.neighbors import KDTree

from metrics import eddies_jaccard_index
from AnDA.AnDA_analog_forecasting import AnDA_analog_forecasting

from tools import (
    grad_desc,
    estimate_sea_level_center,
    compute_u_v,
    get_interpolized_u_v,
    compute_error,
    compute_grad_u_v,
    compute_gradL1,
    convert_from_degree_to_meter,
    convert_from_meter_to_degree,
)
from constants import EQUATORIAL_EARTH_RADIUS

# import netCDF4 as nc
# import scipy.interpolate as sp_interp


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
        coord += coord.real * np.cos(self.mean_pos[1] * np.pi / 180)
        vectors = coord[1:] - coord[:-1]

        # Remove the null vectors
        not_null_vect = vectors != 0
        vectors = vectors[not_null_vect]
        norms = abs(vectors)
        angles = np.angle(vectors[1:] / vectors[:-1])

        if type(delta_time * 1.0) != float:
            delta_time = delta_time[not_null_vect]

        self.coord_list = self.coord_list[np.append(not_null_vect, True), :]
        self.nb_points = len(self.coord_list)

        if self.nb_points <= 2:
            self.winding_angle = 0
            self.angular_velocities = 0
        else:
            # Find the shortest sub streamline with the smallest ratio
            # dist(start,end)/sub_line_len. This ratio will be close to 1 for
            # straight lines and close to 0 for closed loop.

            if cut_lines:
                cumsum_norms = np.concatenate((np.array([0]),np.cumsum(norms)))

                min_ratio = 1

                min_end_id   = self.nb_points - 1
                min_start_id = 0

                for start_id in range(self.nb_points):
                    start_norm = cumsum_norms[start_id]
                    start_pts = coord[start_id]
                    for end_id in range(start_id+1, self.nb_points):
                        curr_len = cumsum_norms[end_id] - start_norm
                        start_end_dist = abs(coord[end_id] - start_pts)
                        ratio = start_end_dist / curr_len
                        if ratio <= min_ratio:
                            min_ratio    = ratio
                            min_end_id   = end_id
                            min_start_id = start_id

                # Recompute the first attributs after taking the sub streamline
                self.coord_list = np.array(
                    self.coord_list[min_start_id : min_end_id + 1], dtype=float
                )
                self.nb_points = (min_end_id - min_start_id) + 1
                self.length = cumsum_norms[end_id] - cumsum_norms[start_id]
                self.mean_pos = np.mean(self.coord_list, axis=0)

                #  Recompute the sub angle list and delta_time
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
        date (int, default=0) : date at which the trajectories are simulated.
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

    def __init__(self, sl_list, date=0, param=[]):
        if len(param) == 0:
            self.date = date
            self.sl_list = list(sl_list)
            self.nb_sl = len(self.sl_list)
            self._set_center()
            self._set_angular_velocity()
            self._set_axis_len_and_dir()

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
        sl_center = np.array(
            [self.sl_list[k].mean_pos for k in range(self.nb_sl)]
        )
        sl_nb_pts = np.array(
            [self.sl_list[k].nb_points for k in range(self.nb_sl)]
        )
        sl_wcenter = [sl_center[k] * sl_nb_pts[k] for k in range(self.nb_sl)]
        self.center = np.sum(sl_wcenter, axis=0) / np.sum(sl_nb_pts)

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

    def _set_axis_len_and_dir(self):
        """Compute the axis direction and lengths for the ellipse

        The eigen vectors are the directions of the axis of the ellipse. The
        length is computed so that sum{(x,y)}{(x/a)**2+(y/b)**2-1} is minimum.

        """
        # Constant
        R = EQUATORIAL_EARTH_RADIUS

        # Compute the covariance matrixe for finding the axis direction
        nb_coord = np.sum(
            [self.sl_list[k].nb_points for k in range(self.nb_sl)]
        )
        merged_coord_list = np.zeros((nb_coord, 2))
        index = 0
        for sl_id in range(self.nb_sl):
            sl = self.sl_list[sl_id]
            merged_coord_list[index : index + sl.nb_points, :] = np.array(
                sl.coord_list
            )
            index += sl.nb_points
        merged_coord_list[:, 0] -= self.center[0]
        merged_coord_list[:, 1] -= self.center[1]
        cov_matrix = np.cov(merged_coord_list.T)
        self.axis_dir = np.linalg.eig(cov_matrix)[1]
        # Rotate and center the points so that the ellipse equation can be
        # written "(x/a)**2 + (y/b)**2 = 1"
        # angles = np.angle(self.axis_dir[:, 0] + 1j * self.axis_dir[:, 1])
        angles = np.angle(self.axis_dir[0, :] + 1j * self.axis_dir[1, :])
        angle = angles[0]

        rotation = np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
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

        # self.axis_len = np.linalg.eig(cov_matrix)[0]
        # Rx, Ry in meter
        axis_len_meter = np.zeros((2,))
        rotation_inv = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        point_rx = np.array([self.axis_len[0], 0])
        point_ry = np.array([0, self.axis_len[1]])
        point_rx = np.dot(rotation_inv, point_rx.T).T
        point_ry = np.dot(rotation_inv, point_ry.T).T
        point_rx_meter = convert_from_degree_to_meter(
            point_rx, self.center[1], R
        )
        point_ry_meter = convert_from_degree_to_meter(
            point_ry, self.center[1], R
        )
        axis_len_meter[0] = np.linalg.norm(point_rx_meter)
        axis_len_meter[1] = np.linalg.norm(point_ry_meter)

        # Esimate h0
        h0 = estimate_sea_level_center(
            self.date,
            self.center,
            aligned_points,
            self.axis_len,
            self.axis_dir,
            self.angular_velocity,
            angle,
            stream_data_fname="../data/data.nc",
            R=EQUATORIAL_EARTH_RADIUS,
        )

        # Compute theorical u,v and get measured (interpolized) u,v
        u_v_evaluated = compute_u_v(
            aligned_points, angle, self.center, axis_len_meter, h0
        )
        u_v_measured = get_interpolized_u_v(
            aligned_points,
            self.date,
            self.center,
            angle,
            stream_data_fname="../data/data.nc",
        )
        # Computing initial error
        L1init = compute_error(u_v_evaluated, u_v_measured)

        # Gradient descent
        step_sizex = axis_len_meter[0]
        step_sizey = axis_len_meter[1]
        n_iter = 10
        for iter in range(n_iter):
            grad_u, grad_v = compute_grad_u_v(
                aligned_points, angle, self.center, axis_len_meter, h0
            )
            gradL1 = compute_gradL1(
                u_v_evaluated, u_v_measured, grad_u, grad_v
            )

            axis_len_meter[0] -= step_sizex * gradL1[0]
            axis_len_meter[1] -= step_sizey * gradL1[1]
            # Updating Rx and Ry in degreee to recompute h0
            point_rx_meter = (
                point_rx_meter
                * axis_len_meter[0]
                / np.linalg.norm(point_rx_meter)
            )
            point_ry_meter = (
                point_ry_meter
                * axis_len_meter[1]
                / np.linalg.norm(point_ry_meter)
            )
            point_rx_degree = convert_from_meter_to_degree(
                point_rx_meter, self.center[1], R
            )
            point_ry_degree = convert_from_meter_to_degree(
                point_ry_meter, self.center[1], R
            )
            self.axis_len[0] = np.linalg.norm(point_rx_degree)
            self.axis_len[1] = np.linalg.norm(point_ry_degree)
            h0 = estimate_sea_level_center(
                self.date,
                self.center,
                aligned_points,
                self.axis_len,
                self.axis_dir,
                self.angular_velocity,
                angle,
                stream_data_fname="../data/data.nc",
                R=EQUATORIAL_EARTH_RADIUS,
            )
            u_v_evaluated = compute_u_v(
                aligned_points, angle, self.center, axis_len_meter, h0
            )
            L1 = compute_error(u_v_evaluated, u_v_measured)
        print("L1 : ", L1)
        print("optimizing: ", L1 <= L1init)

        # Updating Rx,Ry  in degree
        point_rx_meter = (
            point_rx_meter * axis_len_meter[0] / np.linalg.norm(point_rx_meter)
        )
        point_ry_meter = (
            point_ry_meter * axis_len_meter[1] / np.linalg.norm(point_ry_meter)
        )
        point_rx_degree = convert_from_meter_to_degree(
            point_rx_meter, self.center[1], R
        )
        point_ry_degree = convert_from_meter_to_degree(
            point_ry_meter, self.center[1], R
        )
        self.axis_len[0] = np.linalg.norm(point_rx_degree)
        self.axis_len[1] = np.linalg.norm(point_ry_degree)


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


class Observation:
    """Class used to represent observation of eddies and give time scale.

    Args:
        values (array_like) : 6 parameters (axis 1) of different observed eddies
            (axis 0). np.nan if no observed parameter.
        time (array like) : times corresponding to observation and desired
            prediction.
        R (np.array(6,6)) : observation noise covariance matrix.

    Attributes:
        values (array_like) : 6 parameters (axis 1) of different observed eddies
            (axis 0). np.nan if no observed parameter.
        time (array like) : times corresponding to observation and desired
            prediction.
        R (np.array(6,6)) : observation noise covariance matrix.

    """

    def __init__(self, values, time, R=np.ones((6, 6)) / 36):
        self.values = values
        self.time = time
        self.R = R


class ForecastingMethod:
    """Class used to set parameters for the forecasting model.

    Args:
        k (int) : number of analogs to use during the forecast.
        catalog(Catalog) : a catalog of analogs and successors
        regression(String) : chosen regression ('locally_constant', 'increment',
            'local_linear')
    Attributes:
        k (int) : number of analogs to use during the forecast.
        catalog(Catalog) : a catalog of analogs and successors
        regression(String) : chosen regression ('locally_constant', 'increment',
            'local_linear')

    """

    def __init__(self, catalog, k, search, regression="local_linear"):
        self.k = min(k, np.shape(catalog.analogs)[0] - 1)
        self.catalog = catalog
        self.regression = regression
        self.search = search

    def search_neighbors(self, x):
        kdt = KDTree(
            self.catalog.analogs[:, [0, 1]], leaf_size=50, metric="euclidean"
        )
        dist_knn, index_knn = kdt.query(x[:, [0, 1]], self.k)
        if self.search == "complet":
            for i in range(x.shape[0]):
                e1 = Eddy([], None, param=x[i, :])
                for j in range(self.k):
                    e2 = Eddy(
                        [],
                        None,
                        param=self.catalog.analogs[index_knn[i, j], :],
                    )
                    dist_knn[i, j] = dist_knn[i, j] * (
                        1 - eddies_jaccard_index(e1, e2, 10)
                    )
        return dist_knn, index_knn


class FilteringMethod:
    """Class used to set parameters for the filtering model.

    Args:
        method (String) : chosen method ('AnEnKF', 'AnEnKS')
        N (int) : number of members
        R (array_like(6,6)) : observation noise covariance matrix
        forecasting_model (ForecastingModel) : ForecastingModel instance
    Attributes:
        method (String) : chosen method ('AnEnKF', 'AnEnKS')
        N (int) : number of members
        H (array_like(6,6)) : matrix of state transition
        B (array_like(6,6)) : control-input model
        R (array_like(6,6)) : observation noise covariance matrix
        forecasting_model (ForecastingModel) : ForecastingModel instance
        xb (array_like(1,6)) : parameters of the initial eddy

    """

    def __init__(self, R, N, forecasting_method, method="AnEnKS"):
        self.method = method
        self.N = N
        self.xb = None
        self.H = np.eye(6)
        self.B = np.eye(6)
        self.R = R
        self.AF = forecasting_method

    def set_first_eddy(self, xb):
        self.xb = xb

    def m(self, x):
        return AnDA_analog_forecasting(x, self.AF)
