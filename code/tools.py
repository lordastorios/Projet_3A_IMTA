# -*- coding: utf-8 -*-

"""classes.py: This file defines some functions used for eddies detection and
representation."""

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
import netCDF4 as nc
from constants import EQUATORIAL_EARTH_RADIUS
import scipy.interpolate as sp_interp

from constants import EDDY_MIN_AXIS_LEN, EDDY_AXIS_PRECISION


def grad_desc(f_error, f_grad):
    """Compute a local minimum of a function using gradient descente.

    Compute the optimal axis length of the ellipse representing an eddy, using
    an error function and its gradient.

    Args:
        f_error (function(a: float, b: float) = error: float) :
        f_grad (function(a: float, b: float) = gradient: np) :

    Returns:
        a,b (float,float) : The optimal ellipse axis length.

    """

    d0 = 1
    a, b = 1, 1
    epsilon = EDDY_AXIS_PRECISION + 1
    new_err = f_error(a, b)
    while epsilon > EDDY_AXIS_PRECISION:
        da, db = d0, d0

        err_a = new_err
        grad_a = np.sign(f_grad(a, b)[0])
        while (
            f_error(a - grad_a * da, b) > err_a or a - grad_a * da < EDDY_MIN_AXIS_LEN
        ):
            da /= 2.0
        a -= grad_a * da

        err_b = f_error(a, b)
        grad_b = np.sign(f_grad(a, b)[1])
        while (
            f_error(a, b - grad_b * db) > err_b or b - grad_b * db < EDDY_MIN_AXIS_LEN
        ):
            db /= 2.0
        b -= grad_b * db

        new_err = f_error(a, b)
        epsilon = np.sqrt(da * da + db * db)

    return a, b


def pts_is_in_ellipse(pts_list, a, b, angle, center):
    """Test if a point is inside an ellipse.

    Args:
        pts_list (ndarray(,2) of float) : The list of points to be tested.
        a,b (float) : Ellipse axis length.
        angle (float) : Angle between x and a axis.
        center (ndarray(2)): Coordinates of the ellipse center.

    Returns:
        inside (ndarray() of bool) : The test result. The length is the same as
        'pts_list' without the last one.

    """

    centered_pts = np.array(pts_list)
    centered_pts[:, 0] -= center[0]
    centered_pts[:, 1] -= center[1]

    rotation = np.array(
        [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
    )
    aligned_pts = np.dot(rotation, centered_pts.T)

    return (aligned_pts[0, :] ** 2 / a ** 2 + aligned_pts[1, :] ** 2 / b ** 2) <= 1


def estimate_sea_level_center(
    date,
    center_degree,
    aligned_points,
    axis_len,
    axis_dir,
    angular_velocity,
    theta,
    stream_data_fname,
    R=EQUATORIAL_EARTH_RADIUS,
):
    """Return of estimate of sea lever at center of eddy

    Args:
    date : the date on which the eddy is detected
    steam_data_fname : name of the stream data file
    center_degree : coordinate of the center of the eddy in degree
    aligned_points : coordinate in referential of the eddy of all points of all streamlines of the eddy in degree
    axis_len : Rx,Ry
    cov_matrix: covariance matrix of aligned points
    theta : angle of the eddy with the horizontal axis
    R: equatorial earth radius

    returns:
    h0 : sea level at center of eddy (meter)
    """
    # Loading data
    data_set = nc.Dataset(stream_data_fname)
    u_1day = data_set["uo"][date, 0, :]
    v_1day = data_set["vo"][date, 0, :]

    # Data sizes
    # Data step size is 1/12 degree. Due to the C-grid format, an offset of
    #  -0.5*1/12 is added to the axis.
    data_time_size, data_depth_size, data_lat_size, data_lon_size = np.shape(
        data_set["uo"]
    )
    longitudes = np.array(data_set["longitude"]) - 1 / 24
    latitudes = np.array(data_set["latitude"]) - 1 / 24

    # Replace the mask (ie the ground areas) with a null vector field.
    U = np.array(u_1day)
    U[U == -32767] = 0
    V = np.array(v_1day)
    V[V == -32767] = 0

    # Interpolize continuous u,v from the data array
    u_interpolized = sp_interp.RectBivariateSpline(longitudes, latitudes, U.T)
    v_interpolized = sp_interp.RectBivariateSpline(longitudes, latitudes, V.T)

    # Setting coordinate in cartesian space
    rotation_inv = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )

    # Approximation of Rx and Ry in meter
    Rx = axis_len[0] * np.pi * R / 180
    Ry = axis_len[1] * np.pi * R / 180

    # 3rd method to approximate Rx, Ry in meter
    point_rx = np.array([axis_len[0], 0])
    point_ry = np.array([0, axis_len[1]])
    point_rx = np.dot(rotation_inv, point_rx.T).T
    point_ry = np.dot(rotation_inv, point_ry.T).T

    point_rx_meter = convert_from_degree_to_meter(point_rx, center_degree[1], R)
    point_ry_meter = convert_from_degree_to_meter(point_ry, center_degree[1], R)

    Rx2 = np.linalg.norm(point_rx_meter)
    Ry2 = np.linalg.norm(point_ry_meter)

    # For unkown reason, Rx2 and Ry2 aren't equal to Rx,Ry.. error in the convert_from_degree_to_meter function?

    # point to use to compute h0 using max speed: speed is max at Rx or Ry
    xmax = np.array([axis_len[0], 0])
    ymax = np.array([0, axis_len[1]])

    # Setting coordinates in cartesian space in degree

    xmax = np.dot(rotation_inv, xmax.T).T
    ymax = np.dot(rotation_inv, ymax.T).T
    xmax += center_degree
    ymax += center_degree

    # Defining needed constant
    f_corriolis = 10e-4
    g_pesanteur = 9.82

    # We compute h0 using the max value of the current at Ry

    v_max = v_interpolized(ymax[0], ymax[1])
    u_max = u_interpolized(ymax[0], ymax[1])
    max_speed = np.sqrt(v_max ** 2 + u_max ** 2)
    h0 = max_speed * f_corriolis * Ry2 / (g_pesanteur * np.exp(-0.5))

    # if anti clockwise rotation, it is a cold eddy and sea level at center is below zero
    if angular_velocity > 0:
        h0 = -h0

    h0 = float(h0)
    # else, if clockwise rotation, it is a hot eddy and sea level at center is greater than zero
    return h0


def convert_from_degree_to_meter(d_degree, lat, R=EQUATORIAL_EARTH_RADIUS):
    """Convert coordinate in degree to coordinate in meter using relations :
    dy = pi * R * dlat / 180.
    dx = pi * R * dlon / 180. * sin(lat*pi/180)
    relation where dx,dy in meter and dlat,dlon in degrees

    Args:
    d_degree (dlon,dlat): differential of coordinate in degree in cartesian system. shape(2,) if center of an eddy, shape(1,2) otherwise
    lat: latitude of one of the two points on which the differential is computed
    R:equatorial earth radius in meter

    Returns:
    coord_meter: coordinate in meter in cartesian system
    """
    dlon, dlat = d_degree[0], d_degree[1]
    d_meter = np.zeros(np.shape(d_degree))
    if np.shape(d_degree) == (2,):
        dlon, dlat = d_degree[0], d_degree[1]
        d_meter[1] = np.pi * R * d_degree[1] / 180
        d_meter[0] = np.pi * R * d_degree[0] / (180 * np.sin(lat * np.pi / 180))
        d_meter[0] = np.pi * R * d_degree[0] * np.sin(lat * np.pi / 180) / 180
    else:
        dlon, dlat = d_degree[0, 0], d_degree[0, 1]
        d_meter[0, 1] = np.pi * R * d_degree[0, 1] / 180
        d_meter[0, 0] = np.pi * R * d_degree[0, 0] / (180 * np.sin(lat * np.pi / 180))
        d_meter[0, 0] = np.pi * R * d_degree[0, 0] * np.sin(lat * np.pi / 180) / 180

    return d_meter


def compute_u_v(point, theta, center_degree, axis_len, h0, R=EQUATORIAL_EARTH_RADIUS):
    """Compute the theorical value of u and v at a given point

    Args:
    point: coordinate of point in the referential of the eddy in degree
    theta : angle of the eddy with the horizontal axis
    center_degree: coordinates of the center of the eddy in degree
    Rx : value of the axes length along axis x in eddy referential
    Ry : value of the axes length along axis y in eddy referential
    h0 : sea level at the center of the eddy,

    WARNING : x must be along the big axis and y along the small axis
    Returns:
    u : zonal current at the point
    v : meridonal current at the point
    """
    # Defining needed constant
    f_corriolis = 10e-4
    rho_sea = 1030
    g_pesanteur = 9.82

    # rotation matrix

    rotation_inv = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    # Setting coordinates in cartesian space
    point = np.dot(rotation_inv, point.T).T
    point += center_degree

    # Setting coordinates  in meter
    point_meter = convert_from_degree_to_meter(point, point[1])
    center_meter = convert_from_degree_to_meter(center_degree, center_degree[1])
    X, Y = point_meter[0], point_meter[1]
    X0, Y0 = center_meter[0], center_meter[1]

    # Approximation of Rx and Ry in meter
    Rx = axis_len[0] * np.pi * R / 180
    Ry = axis_len[1] * np.pi * R / 180

    # 3rd method to approximate Rx, Ry in meter
    point_rx = np.array([axis_len[0], 0])
    point_ry = np.array([0, axis_len[1]])
    point_rx = np.dot(rotation_inv, point_rx.T).T
    point_ry = np.dot(rotation_inv, point_ry.T).T

    point_rx_meter = convert_from_degree_to_meter(point_rx, center_degree[1], R)
    point_ry_meter = convert_from_degree_to_meter(point_ry, center_degree[1], R)

    Rx2 = np.linalg.norm(point_rx_meter)
    Ry2 = np.linalg.norm(point_ry_meter)

    # This definition of Rx allow to get a value for u and v but with it we find a value of ~20m for h0
    # constant to compute sea level
    a = np.cos(theta) ** 2 / (2 * Rx2 ** 2) + np.sin(theta) ** 2 / (2 * Ry2 ** 2)
    b = -np.sin(2 * theta) / (4 * Rx2 ** 2) + np.sin(2 * theta) / (4 * Ry2 ** 2)
    c = np.sin(theta) ** 2 / (2 * Rx2 ** 2) + np.cos(theta) ** 2 / (2 * Ry2 ** 2)

    # Computing sea level
    sea_level = h0 * np.exp(
        -(a * (X - X0) ** 2 - 2 * b * (X - X0) * (Y - Y0) + c * (Y - Y0) ** 2)
    )
    # Computing value of u and v at point (X,Y)
    u = -g_pesanteur * 2 * (b * (X - X0) - c * (Y - Y0)) * sea_level / f_corriolis
    v = g_pesanteur * 2 * (b * (Y - Y0) - a * (X - X0)) * sea_level / f_corriolis
    u_v = np.array([u, v])
    return u_v


def get_interpolized_u_v(point, date, center_degree, theta, stream_data_fname):
    """Get the interpolized value of u and v at point
    Args:
    point: coordinate of point in the referential of the eddy in degree
    date : the date on which the eddy is detected
    center_degree : coordinate of center in degree
    steam_data_fname : name of the stream data file
    theta: angle of the eddy
    """
    # Loading data
    data_set = nc.Dataset(stream_data_fname)
    u_1day = data_set["uo"][date, 0, :]
    v_1day = data_set["vo"][date, 0, :]

    # Data sizes
    # Data step size is 1/12 degree. Due to the C-grid format, an offset of
    #  -0.5*1/12 is added to the axis.
    data_time_size, data_depth_size, data_lat_size, data_lon_size = np.shape(
        data_set["uo"]
    )
    longitudes = np.array(data_set["longitude"]) - 1 / 24
    latitudes = np.array(data_set["latitude"]) - 1 / 24

    # Replace the mask (ie the ground areas) with a null vector field.
    U = np.array(u_1day)
    U[U == -32767] = 0
    V = np.array(v_1day)
    V[V == -32767] = 0

    # Interpolize continuous u,v from the data array
    u_interpolized = sp_interp.RectBivariateSpline(longitudes, latitudes, U.T)
    v_interpolized = sp_interp.RectBivariateSpline(longitudes, latitudes, V.T)

    # Setting coordinate in degree in cartesian space
    rotation_inv = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    point = np.dot(rotation_inv, point.T).T
    point += center_degree
    u_measured = u_interpolized(point[0], point[1])
    v_measured = v_interpolized(point[0], point[1])
    u_v_measured = np.array([float(u_measured), float(v_measured)])
    return u_v_measured


def compute_error(points, Rx, Ry, h0, data):
    """Compute the error between theorical value of u and v and measured values

    Args:
    data : u and v measured value
    """