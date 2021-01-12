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
    h0 : sea level at center of eddy (unity??)
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
    coordinate_cartesian = np.dot(rotation_inv, aligned_points.T).T
    coordinate_cartesian[:, 0] += center_degree[0]
    coordinate_cartesian[:, 1] += center_degree[1]

    # Setting coordinate in meter
    center_meter = convert_from_degree_to_meter(center_degree, R)
    coordinate_cartesian_in_meter = np.zeros(np.shape(coordinate_cartesian))
    for point in range(len(coordinate_cartesian_in_meter)):
        coordinate_cartesian_in_meter[point] = convert_from_degree_to_meter(
            coordinate_cartesian[point], R
        )

    # First approximation of R_x and R_y in meter
    R_x = axis_len[0] * np.pi * R / 180
    R_y = axis_len[1] * np.pi * R / 180
    print("center: ", center_degree)

    # Zone of interest for sea_level_center estimation (points on axes of eddy )
    """
    #zone of interest for first approach
    nb_points_estimation = 50
    x_interest = np.zeros((nb_points_estimation, 2))
    x_interest[:, 0] = np.linspace(
        np.min(aligned_points[:, 0]),
        np.max(aligned_points[:, 0]),
        num=nb_points_estimation,
    )
    y_interest = np.zeros((nb_points_estimation, 2))
    y_interest[:, 1] = np.linspace(
        np.min(aligned_points[:, 1]),
        np.max(aligned_points[:, 1]),
        num=nb_points_estimation,
    )
    """
    # Zone of interest for 2nd approach
    nb_points_estimation = 50
    x_interest = np.zeros((nb_points_estimation, 2))
    x_interest[:, 0] = np.linspace(
        axis_len[0] / np.sqrt(2),
        axis_len[0] / np.sqrt(2),
        num=nb_points_estimation,
    )
    y_interest = np.zeros((nb_points_estimation, 2))
    y_interest[:, 1] = np.linspace(
        axis_len[1] / np.sqrt(2),
        axis_len[1] / np.sqrt(2),
        num=nb_points_estimation,
    )

    # Setting point of interest for sea level estimation in correct coordiates

    # Setting coordinates in cartesian space in degree
    x_interest = np.dot(rotation_inv, x_interest.T).T
    y_interest = np.dot(rotation_inv, y_interest.T).T
    x_interest[:, 0] += center_degree[0]
    x_interest[:, 1] += center_degree[1]
    y_interest[:, 0] += center_degree[0]
    y_interest[:, 1] += center_degree[1]
    # Setting coordinates in meter
    set_interest1 = np.zeros(np.shape(x_interest))
    set_interest2 = np.zeros(np.shape(x_interest))
    for point in range(len(set_interest1)):
        set_interest1[point] = convert_from_degree_to_meter(x_interest[point], R)
    for point in range(len(set_interest2)):
        set_interest2[point] = convert_from_degree_to_meter(y_interest[point], R)

    # Defining needed constant
    f_corriolis = 10e-4
    g_pesanteur = 9.82
    a = np.cos(theta) ** 2 / (2 * R_x ** 2) + np.sin(theta) ** 2 / (2 * R_y ** 2)
    b = -np.sin(2 * theta) / (4 * R_x ** 2) + np.sin(2 * theta) / (4 * R_y ** 2)
    c = np.sin(theta) ** 2 / (2 * R_x ** 2) + np.cos(theta) ** 2 / (2 * R_y ** 2)

    # Computing sea_level at center of eddy

    """ First approach, using derivation of the level of sea and the measured current
    # Considering axis x of eddy ,deriving with respect to x

    h0_x = 0
    for point in range(len(set_interest1)):
        factor = np.exp(
            -(
                a * (set_interest1[point, 0] - center_meter[0]) ** 2
                - 2
                * b
                * (set_interest1[point, 0] - center_meter[0])
                * (set_interest1[point, 1] - center_meter[1])
                + c * (set_interest1[point, 1] - center_meter[1]) ** 2
            )
        )
        factor = (
            factor
            * g_pesanteur
            * (
                -2 * a * (set_interest1[point, 0] - center_meter[0])
                + 2 * b * (set_interest1[point, 1] - center_meter[1])
            )
            / f_corriolis
        )
        h_approx = v_interpolized(x_interest[point, 0], x_interest[point, 1]) / factor
        h0_x += h_approx
    h0_x /= len(set_interest1)

    # Deriving with respect to y
    h0_x = 0
    for point in range(len(set_interest1)):
        factor = np.exp(
            -(
                a * (set_interest1[point, 0] - center_meter[0]) ** 2
                - 2
                * b
                * (set_interest1[point, 0] - center_meter[0])
                * (set_interest1[point, 1] - center_meter[1])
                + c * (set_interest1[point, 1] - center_meter[1]) ** 2
            )
        )
        factor = (
            factor
            * g_pesanteur
            * (
                2 * b * (set_interest1[point, 0] - center_meter[0])
                - 2 * c * (set_interest1[point, 1] - center_meter[1])
            )
            / f_corriolis
        )
        h_approx = u_interpolized(x_interest[point, 0], x_interest[point, 1]) / factor
        h0_x += h_approx
    h0_x /= len(set_interest1)
    h0 = h0_x

    For unkown reason this approach doesn't work now
    """

    # 2nd approach
    """
    we make the hypothesis of a circular eddy 2 consecutive time with R=Rx and Ry to compute h0 and we take the average
    In a circular eddy, the max of current is at r=R/sqrt(2)
    """
    h0_x = 0
    for point in range(len(set_interest1)):
        v = v_interpolized(x_interest[point, 0], x_interest[point, 1])
        u = u_interpolized(x_interest[point, 0], x_interest[point, 1])
        speed = np.sqrt(v ** 2 + u ** 2)
        hmax = speed * f_corriolis * R_x / (g_pesanteur * np.exp(-0.5))
        h0_x += hmax
    h0_x /= len(set_interest1)

    h0_y = 0
    for point in range(len(set_interest2)):
        v_max = v_interpolized(y_interest[point, 0], y_interest[point, 1])
        u_max = u_interpolized(y_interest[point, 0], y_interest[point, 1])
        max_speed = np.sqrt(v_max ** 2 + u_max ** 2)
        hmax = max_speed * f_corriolis * R_y / (g_pesanteur * np.exp(-0.5))
        h0_y += hmax
    h0_y /= len(set_interest2)
    h0 = (h0_x + h0_y) / 2

    return h0


def convert_from_degree_to_meter(coord_degree, R=EQUATORIAL_EARTH_RADIUS):
    """Convert coordinate in degree to coordinate in meter using relations :
    dy = pi * R * dlat / 180.
    dx = pi * R * dlon / 180. * sin(lat*pi/180)
    relation where dx,dy in meter and dlat,dlon in degrees

    Args:
    coord_degree (lon,lat): coordinate in degree in cartesian system. shape(2,) if center of an eddy, shape(1,2) otherwise
    R:equatorial earth radius in meter

    Returns:
    coord_meter: coordinate in meter in cartesian system
    """
    lon, lat = coord_degree[0], coord_degree[1]
    coord_meter = np.zeros(np.shape(coord_degree))
    if np.shape(coord_degree) == (2,):
        lon, lat = coord_degree[0], coord_degree[1]
        coord_meter[1] = np.pi * R * coord_degree[1] / 180
        coord_meter[0] = np.pi * R * coord_degree[0] / (180 * np.sin(lat * np.pi / 180))
    else:
        lon, lat = coord_degree[0, 0], coord_degree[0, 1]
        coord_meter[0, 1] = np.pi * R * coord_degree[0, 1] / 180
        coord_meter[0, 0] = (
            np.pi * R * coord_degree[0, 0] / (180 * np.sin(lat * np.pi / 180))
        )
    return coord_meter


def compute_u_v(point, theta, sigma_x, sigma_y, sea_level_center):
    """Compute the theorical value of u and v at a given point

    Args:
    point: coordinate of point in the referential of the eddy
    theta : angle of the eddy with the horizontal axis
    sigma_x : value of the axes length along axis x in eddy referential
    sigma_y : value of the axes length along axis y in eddy referential
    sea_level_center : sea level at the center of the eddy,

    WARNING : x must be along the big axis and y along the small axis
    Returns:
    u : zonal current at the point
    v : meridonal current at the point
    """
    # Defining needed constant
    f_corriolis = 10e-4
    rho_sea = 1030

    # Coordinate in referential of the eddy
    x = pt[0]
    y = pt[1]
    # Defining coordinate in cartesian coordinate
    X = x * np.cos(theta) - y * np.sin(theta)
    Y = x * np.sin(theta) + y * np.cos(theta)
    sea_level = (
        sea_level_center
        * exp(-(X ** 2) / (2 * sigma_x ** 2))
        * exp(-(Y ** 2) / (2 * sigma_y ** 2))
    )
    # Computing value of u and v at point (X,Y)
    u = (
        (f_corriolis / rho_sea)
        * (X / sigma_x ** 2)
        * (np.cos(theta) - np.sin(theta))
        * sea_level
    )
    v = (
        -(f_corriolis / rho_sea)
        * (Y / sigma_y ** 2)
        * (np.sin(theta) + np.cos(theta))
        * sea_level
    )

    return (u, v)


def compute_error(points, sigma_x, sigma_y, sea_level_center, data):
    """Compute the error between theorical value of u and v and measured values

    Args:
    data : u and v measured value
    """