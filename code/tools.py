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

    # Rx, Ry in meter
    point_rx = np.array([axis_len[0], 0])
    point_ry = np.array([0, axis_len[1]])
    point_rx = np.dot(rotation_inv, point_rx.T).T
    point_ry = np.dot(rotation_inv, point_ry.T).T

    point_rx_meter = convert_from_degree_to_meter(point_rx, center_degree[1], R)
    point_ry_meter = convert_from_degree_to_meter(point_ry, center_degree[1], R)

    Rx = np.linalg.norm(point_rx_meter)
    Ry = np.linalg.norm(point_ry_meter)

    # Rx = axis_len[0]
    # Ry = axis_len[1]

    # point to use to compute h0 using max speed: speed is max at Rx or Ry
    ymax = np.array([0, axis_len[1]])

    # Setting coordinates in cartesian space in degree
    ymax = np.dot(rotation_inv, ymax.T).T
    ymax += center_degree

    # Defining needed constant
    f_corriolis = 10e-4
    g_pesanteur = 9.82

    # We compute h0 using the max value of the current at Ry
    v_max = v_interpolized(ymax[0], ymax[1])
    u_max = u_interpolized(ymax[0], ymax[1])
    max_speed = np.sqrt(v_max ** 2 + u_max ** 2)
    h0 = max_speed * f_corriolis * Ry / (g_pesanteur * np.exp(-0.5))

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
    d_meter: differential in meter in cartesian system
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


def convert_from_meter_to_degree(d_meter, lat, R=EQUATORIAL_EARTH_RADIUS):
    """Convert a differential of coordinates from meter to dregree
    d_meter=(dx,dy)
    lat: latitude of one of the two points on which the differential is computed
    """
    dx, dy = d_meter[0], d_meter[1]
    d_degree = np.zeros(np.shape(d_meter))
    if np.shape(d_degree) == (2,):
        dx, dy = d_meter[0], d_meter[1]
        d_degree[1] = d_meter[1] * 180 / (np.pi * R)
        d_degree[0] = d_meter[0] * 180 / (np.sin(lat * np.pi / 180) * np.pi * R)
    else:
        dx, dy = d_meter[0, 0], d_meter[0, 1]
        d_degree[0, 1] = d_meter[0, 1] * 180 / (np.pi * R)
        d_degree[0, 0] = d_degree[0, 0] * 180 / (np.sin(lat * np.pi / 180) * np.pi * R)
    return d_degree


def compute_u_v(points, theta, center_degree, axis_len, h0, R=EQUATORIAL_EARTH_RADIUS):
    """Compute the theorical value of u and v at a given point

    Args:
    points: ndarray(n,2) coordinate of points in the referential of the eddy in degree
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
    """
    # Rx , Ry in meter
    point_rx = np.array([axis_len[0], 0])
    point_ry = np.array([0, axis_len[1]])
    point_rx = np.dot(rotation_inv, point_rx.T).T
    point_ry = np.dot(rotation_inv, point_ry.T).T

    point_rx_meter = convert_from_degree_to_meter(point_rx, center_degree[1], R)
    point_ry_meter = convert_from_degree_to_meter(point_ry, center_degree[1], R)

    Rx = np.linalg.norm(point_rx_meter)
    Ry = np.linalg.norm(point_ry_meter)
    """
    Rx = axis_len[0]
    Ry = axis_len[1]
    # constant to compute sea level
    a = np.cos(theta) ** 2 / (2 * Rx ** 2) + np.sin(theta) ** 2 / (2 * Ry ** 2)
    b = -np.sin(2 * theta) / (4 * Rx ** 2) + np.sin(2 * theta) / (4 * Ry ** 2)
    c = np.sin(theta) ** 2 / (2 * Rx ** 2) + np.cos(theta) ** 2 / (2 * Ry ** 2)

    # Initializing u_v
    n = len(points)
    u_v = np.zeros((n, 2))
    # Setting coordinates in cartesian space
    points = np.dot(rotation_inv, points.T).T
    points_meter = np.zeros((n, 2))
    for i in range(n):
        points_meter[i] = convert_from_degree_to_meter(points[i], center_degree[1], R)

    dX, dY = points_meter[:, 0], points_meter[:, 1]

    # Computing sea level
    for i in range(n):
        sea_level = h0 * np.exp(
            -(a * (dX[i]) ** 2 - 2 * b * (dX[i]) * (dY[i]) + c * (dY[i]) ** 2)
        )
        u_v[i, 0] = (
            -g_pesanteur * 2 * (b * (dX[i]) - c * (dY[i])) * sea_level / f_corriolis
        )
        u_v[i, 1] = (
            g_pesanteur * 2 * (b * (dY[i]) - a * (dX[i])) * sea_level / f_corriolis
        )

    return u_v


def get_interpolized_u_v(points, date, center_degree, theta, stream_data_fname):
    """Get the interpolized value of u and v at point
    Args:
    point: ndarray(n,2) coordinate of point in the referential of the eddy in degree
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
    points = np.dot(rotation_inv, points.T).T
    points += center_degree

    n = len(points)  # nbr of points
    u_v_measured = np.zeros((n, 2))
    for i in range(n):
        u_v_measured[i, 0] = float(u_interpolized(points[i, 0], points[i, 1]))
        u_v_measured[i, 1] = float(v_interpolized(points[i, 0], points[i, 1]))
    return u_v_measured


def compute_error(u_v_evaluated, u_v_measured):
    """Compute the error between theorical value of u and v and measured values

    Args:
    point: on which to compute the error
    u_v_evaluated: ndarray(1,2): thorical [u,v] values computed
    u_v_measure: ndarray(1,2): measured [u,v] values

    Return:
    L1 error
    L2 error
    """
    L1 = np.zeros(len(u_v_evaluated))
    for i in range(len(u_v_evaluated)):
        L1[i] = (u_v_evaluated[i, 0] - u_v_measured[i, 0]) ** 2 + (
            u_v_evaluated[i, 1] - u_v_measured[i, 1]
        ) ** 2
    L1 = np.sum(L1)
    return L1


def compute_grad_u_v(
    points, theta, center_degree, axis_len, h0, R=EQUATORIAL_EARTH_RADIUS
):
    """Compute the gradient of u_evaluated and v_evaluated with respect to Rx and Ry"""
    # Defining needed constant
    f_corriolis = 10e-4
    rho_sea = 1030
    g_pesanteur = 9.82

    # rotation matrix

    rotation_inv = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    """
    # approximate Rx, Ry in meter
    point_rx = np.array([axis_len[0], 0])
    point_ry = np.array([0, axis_len[1]])
    point_rx = np.dot(rotation_inv, point_rx.T).T
    point_ry = np.dot(rotation_inv, point_ry.T).T

    point_rx_meter = convert_from_degree_to_meter(point_rx, center_degree[1], R)
    point_ry_meter = convert_from_degree_to_meter(point_ry, center_degree[1], R)

    Rx = np.linalg.norm(point_rx_meter)
    Ry = np.linalg.norm(point_ry_meter)
    """
    Rx = axis_len[0]
    Ry = axis_len[1]

    # constant to compute sea level
    a = np.cos(theta) ** 2 / (2 * Rx ** 2) + np.sin(theta) ** 2 / (2 * Ry ** 2)
    b = -np.sin(2 * theta) / (4 * Rx ** 2) + np.sin(2 * theta) / (4 * Ry ** 2)
    c = np.sin(theta) ** 2 / (2 * Rx ** 2) + np.cos(theta) ** 2 / (2 * Ry ** 2)

    # Initializing u_v
    n = len(points)
    grad_u = np.zeros((n, 2))
    grad_v = np.zeros((n, 2))

    # Setting coordinates in cartesian space
    points = np.dot(rotation_inv, points.T).T
    points_meter = np.zeros((n, 2))
    for i in range(n):
        points_meter[i] = convert_from_degree_to_meter(points[i], center_degree[1], R)

    dX, dY = points_meter[:, 0], points_meter[:, 1]

    # Computing sea level
    for i in range(n):
        sea_level = h0 * np.exp(
            -(a * (dX[i]) ** 2 - 2 * b * (dX[i]) * (dY[i]) + c * (dY[i]) ** 2)
        )
        h = -g_pesanteur * 2 * (b * (dX[i]) - c * (dY[i])) / f_corriolis
        g = g_pesanteur * 2 * (b * (dY[i]) - a * (dX[i])) / f_corriolis
        hprime_x = (
            -2
            * g_pesanteur
            * (
                np.sin(2 * theta) * dX[i] / (2 * Rx ** 3)
                + (np.sin(theta) ** 2) * dY[i] / (Rx ** 3)
            )
        )
        hprime_y = (
            -2
            * g_pesanteur
            * (
                -np.sin(2 * theta) * dX[i] / (2 * Ry ** 3)
                + (np.cos(theta) ** 2) * dY[i] / (Ry ** 3)
            )
        )
        gprime_x = (
            -2
            * g_pesanteur
            * (
                np.sin(2 * theta) * dY[i] / (2 * Rx ** 3)
                + (np.cos(theta) ** 2) * dX[i] / (Rx ** 3)
            )
        )
        gprime_y = (
            -2
            * g_pesanteur
            * (
                -np.sin(2 * theta) * dY[i] / (2 * Ry ** 3)
                + (np.sin(theta) ** 2) * dX[i] / (Ry ** 3)
            )
        )
        sea_level_prime_x = (
            (
                (np.cos(theta) * dX[i]) ** 2
                + np.sin(2 * theta) * dX[i] * dY[i]
                + (np.sin(theta) * dY[i]) ** 2
            )
            * sea_level
            / (Rx ** 3)
        ) * sea_level
        sea_level_prime_y = (
            (
                (np.sin(theta) * dX[i]) ** 2
                - np.sin(2 * theta) * dX[i] * dY[i]
                + (np.cos(theta) * dY[i]) ** 2
            )
            * sea_level
            / (Ry ** 3)
        ) * sea_level
        grad_u[i, 0] = hprime_x * sea_level + h * sea_level_prime_x
        grad_u[i, 1] = hprime_y * sea_level + h * sea_level_prime_y
        grad_v[i, 0] = gprime_x * sea_level + g * sea_level_prime_x
        grad_v[i, 1] = gprime_y * sea_level + g * sea_level_prime_y

    return grad_u, grad_v


def compute_gradL1(u_v_evaluated, u_v_measured, grad_u, grad_v):
    """Compute the gradient of L1 with respect to Rx and Ry"""
    gradL1 = np.zeros((2, 1))
    n = len(u_v_evaluated)
    gradL1_x = 2 * np.sum(
        (u_v_evaluated[:, 0] - u_v_measured[:, 0]) * grad_u[:, 0]
        + (u_v_evaluated[:, 1] - u_v_measured[:, 1]) * grad_v[:, 0]
    )
    gradL1_y = 2 * np.sum(
        (u_v_evaluated[:, 0] - u_v_measured[:, 0]) * grad_u[:, 1]
        + (u_v_evaluated[:, 1] - u_v_measured[:, 1]) * grad_v[:, 1]
    )
    gradL1[0] = gradL1_x
    gradL1[1] = gradL1_y

    return gradL1
