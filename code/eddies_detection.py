# -*- coding: utf-8 -*-

"""classes.py: This file defines the functions used for streamlines
computation, processing and clustering and eddies detection."""

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


import netCDF4 as nc
import numpy as np
import os
import scipy.integrate as sp_integ
import scipy.interpolate as sp_interp

from parcels import FieldSet, ParticleSet, ScipyParticle, ErrorCode
from parcels import AdvectionAnalytical, AdvectionRK4
from datetime import timedelta as delta

from classes import StreamLine, Eddy
from constants import EQUATORIAL_EARTH_RADIUS, MIN_STREAMLINE_IN_EDDIES
from constants import EDDY_MIN_RADIUS


def get_traj_with_parcels(
    date, runtime, delta_time, particle_grid_step, stream_data_fname
):
    """Compute trajectories of particles in the sea using parcels library.

    Compute trajectories of particles in the sea at a given date, in a static 2D
    field of stream using Runge Kutta 4th order algorithm.

    Args:
        date (int) : Day in number of days relatively to the data time origin at
            which the stream data should be taken.
        runtime (int) : Total duration in hours of the field integration.
            Trajectories length increases with the runtime.
        delta_time (int) : Time step in hours of the integration.
        particle_grid_step (int) : Grid step size for the initial positions of
            the particles. The unit is the data index step, ie data dx and dy.
        stream_data_fname (str) : Complete name of the stream data file.

    Returns:
        stream_line_list (list of classes.StreamLine) : The list of trajectories.

    Notes:
        The input file is expected to contain the daily mean fields of east- and
        northward ocean current velocity (uo,vo) in a format as described here:
        http://marine.copernicus.eu/documents/PUM/CMEMS-GLO-PUM-001-024.pdf.


    """

    # Loading data
    data_set = nc.Dataset(stream_data_fname)
    u_1day = data_set["uo"][date, 0, :]
    v_1day = data_set["vo"][date, 0, :]

    # Data sizes
    data_time_size, data_depth_size, data_lat_size, data_lon_size = np.shape(
        data_set["uo"]
    )
    longitudes = np.array(data_set["longitude"])
    latitudes = np.array(data_set["latitude"])

    # Replace the mask (ie the ground areas) with a null vector field.
    U = np.array(u_1day)
    U[U == -32767] = 0
    V = np.array(v_1day)
    V[V == -32767] = 0

    #  Initialize a field set using the data set.
    data = {"U": U, "V": V}
    dimensions = {"lon": longitudes, "lat": latitudes}
    fieldset = FieldSet.from_data(data, dimensions, mesh="spherical")
    fieldset.U.interp_method = "cgrid_velocity"
    fieldset.V.interp_method = "cgrid_velocity"

    # List of initial positions of the particles. Particles on the ground are
    # removed.
    init_pos = []
    for lon in range(0, data_lon_size, particle_grid_step):
        for lat in range(0, data_lat_size, particle_grid_step):
            if not u_1day[lat, lon] is np.ma.masked:
                init_pos.append([longitudes[lon], latitudes[lat]])
    init_pos = np.array(init_pos)
    init_longitudes = init_pos[:, 0]
    init_latitudes = init_pos[:, 1]

    #  Initialize particle set
    pSet = ParticleSet.from_list(
        fieldset, ScipyParticle, init_longitudes, init_latitudes, depth=None, time=None
    )

    # Initialize output file and run simulation
    def DeleteParticle(particle, fieldset, time):
        particle.delete()

    output = pSet.ParticleFile(
        name="trajectories_temp.nc", outputdt=delta(hours=delta_time)
    )
    pSet.execute(
        AdvectionRK4,
        runtime=delta(hours=runtime),
        dt=np.inf,
        output_file=output,
        recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle},
    )
    output.close()

    # Load simulation results and create the stream line list
    nc_traj = nc.Dataset("trajectories_temp.nc")
    trajectories = np.zeros(
        (nc_traj.dimensions["traj"].size, nc_traj.dimensions["obs"].size, 2)
    )
    trajectories[:, :, 0] = np.array(nc_traj["lon"])
    trajectories[:, :, 1] = np.array(nc_traj["lat"])
    stream_line_list = []
    for trajectory in trajectories:
        stream_line_list.append(
            StreamLine(trajectory[np.isfinite(trajectory[:, 0])], delta_time * 3600)
        )

    # Clean working dir
    os.system("rm -r trajectories_temp*")

    return stream_line_list


def get_traj_with_scipy(
    date,
    runtime,
    max_delta_time,
    particle_grid_step,
    stream_data_fname,
    R=EQUATORIAL_EARTH_RADIUS,
):
    """Compute trajectories of particles in the sea using scipy library.

    Compute trajectories of particles in the sea at a given date, in a static 2D
    field of stream using Runge Kutta 4th order algorithm.

    Args:
        date (int) : Day in number of days relatively to the data time origin at
            which the stream data should be taken.
        runtime (int) : Total duration in hours of the field integration.
            Trajectories length increases with the runtime.
        max_delta_time (int) : Maximum time step in hours of the integration.
            The integration function used can use smaller time step if needed.
        particle_grid_step (int) : Grid step size for the initial positions of
            the particles. The unit is the data index step, ie data dx and dy.
        stream_data_fname (str) : Complete name of the stream data file.
        R (float) : Radius of the Earth in meter in the data area. The exact
            value can be used to reduce Earth shape related conversion and
            computation error. Default is the equatorial radius.

    Returns:
        stream_line_list (list of classes.StreamLine) : The list of trajectories.

    Notes:
        The input file is expected to contain the daily mean fields of east- and
        northward ocean current velocity (uo,vo) in a format as described here:
        http://marine.copernicus.eu/documents/PUM/CMEMS-GLO-PUM-001-024.pdf.

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

    # Conversion of u and v from m.s**-1 to degree.s**-1
    conversion_coeff = 360 / (2 * np.pi * R)
    U *= conversion_coeff
    V *= conversion_coeff
    for lat in range(data_lat_size):
        U[lat, :] *= 1 / np.cos(latitudes[lat] / 360)

    # Interpolize continuous u,v from the data array
    u_interpolized = sp_interp.RectBivariateSpline(longitudes, latitudes, U.T)
    v_interpolized = sp_interp.RectBivariateSpline(longitudes, latitudes, V.T)

    def stream(t, pos_vect):
        # pos = (long,lat)
        u = u_interpolized.ev(pos_vect[0], pos_vect[1])
        v = v_interpolized.ev(pos_vect[0], pos_vect[1])
        return np.array([u, v])

    # List of initial positions of the particles. Particles on the ground are
    # removed.
    init_pos = []
    for lon in range(0, data_lon_size, particle_grid_step):
        for lat in range(0, data_lat_size, particle_grid_step):
            if not u_1day[lat, lon] is np.ma.masked:
                init_pos.append([longitudes[lon], latitudes[lat]])
    init_pos = np.array(init_pos)
    nb_stream_line = len(init_pos)

    # Integrate each positions
    stream_line_list = []
    progression = 0
    print("Integration: ", end="", sep="")
    for sl_id in range(nb_stream_line):
        if int(sl_id / nb_stream_line * 100) - progression >= 5:
            progression += 5
            print(progression, "% ", end="", sep="")

        coord_list = [np.array(init_pos[sl_id])]
        dt_list = []
        previous_t = 0
        integration_instance = sp_integ.RK45(
            stream,
            0,
            init_pos[sl_id],
            runtime * 3600,
            rtol=1e-4,
            atol=1e-7,
            max_step=max_delta_time * 3600,
        )
        # Integrate the trajectory dt by dt
        while integration_instance.status == "running":
            previous_pt = coord_list[-1]
            is_between_limit_lat = latitudes[0] <= previous_pt[1] <= latitudes[-1]
            is_between_limit_long = longitudes[0] <= previous_pt[0] <= longitudes[-1]
            if is_between_limit_lat and is_between_limit_long:
                integration_instance.step()
                coord_list.append(integration_instance.y)
            else:
                integration_instance.status = "finished"
                coord_list.append(previous_pt)
            dt_list.append(integration_instance.t - previous_t)
            previous_t = integration_instance.t

        dt_list = np.array(dt_list)
        stream_line_list.append(StreamLine(coord_list, dt_list))
    print()

    return stream_line_list


def get_traj_with_numpy(
    date,
    runtime,
    delta_time,
    particle_grid_step,
    stream_data_fname,
    R=EQUATORIAL_EARTH_RADIUS,
):
    """Compute trajectories of particles in the sea using only numpy library.

    Compute trajectories of particles in the sea at a given date, in a static 2D
    field of stream using Runge Kutta 4th order algorithm.

    Args:
        date (int) : Day in number of days relatively to the data time origin at
            which the stream data should be taken.
        runtime (int) : Total duration in hours of the field integration.
            Trajectories length increases with the runtime.
        delta_time (int) : Maximum time step in hours of the integration.
            The integration function used can use smaller time step if needed.
        particle_grid_step (int) : Grid step size for the initial positions of
            the particles. The unit is the data index step, ie data dx and dy.
        stream_data_fname (str) : Complete name of the stream data file.
        R (float) : Radius of the Earth in meter in the data area. The exact
            value can be used to reduce Earth shape related conversion and
            computation error. Default is the equatorial radius.

    Returns:
        stream_line_list (list of classes.StreamLine) : The list of trajectories.

    Notes:
        The input file is expected to contain the daily mean fields of east- and
        northward ocean current velocity (uo,vo) in a format as described here:
        http://marine.copernicus.eu/documents/PUM/CMEMS-GLO-PUM-001-024.pdf.

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

    # Conversion of u and v from m.s**-1 to degree.s**-1
    conversion_coeff = 360 / (2 * np.pi * R)
    U *= conversion_coeff
    V *= conversion_coeff
    for lat in range(data_lat_size):
        U[lat, :] *= 1 / np.cos(latitudes[lat] / 360)

    # Interpolize continuous u,v from the data array
    u_interpolized = sp_interp.RectBivariateSpline(longitudes, latitudes, U.T)
    v_interpolized = sp_interp.RectBivariateSpline(longitudes, latitudes, V.T)

    def stream(pos_vect):
        # pos = (long,lat)
        u = u_interpolized.ev(pos_vect[0], pos_vect[1])
        v = v_interpolized.ev(pos_vect[0], pos_vect[1])
        return np.array([u, v])

    # List of initial positions of the particles. Particles on the ground are
    # removed.
    init_pos = []
    for lon in range(0, data_lon_size, particle_grid_step):
        for lat in range(0, data_lat_size, particle_grid_step):
            if not u_1day[lat, lon] is np.ma.masked:
                init_pos.append([longitudes[lon], latitudes[lat]])
    init_pos = np.array(init_pos)
    nb_sl = len(init_pos)

    # Integration dt by dt, over the 'runtime' duration
    dt = 3600 * delta_time
    nb_step = int(runtime / delta_time)
    sl_array = np.zeros((nb_step + 1, nb_sl, 2))
    sl_array[0, :, :] = np.array(init_pos)

    def rk_4(pos):
        if np.isnan(pos[0]):
            return np.nan * np.zeros(2)
        if not (latitudes[0] <= pos[1] <= latitudes[-1]):
            return np.nan * np.zeros(2)
        if not (longitudes[0] <= pos[0] <= longitudes[-1]):
            return np.nan * np.zeros(2)
        k1 = dt * stream(pos)
        k2 = dt * stream(pos + k1 * 0.5)
        k3 = dt * stream(pos + k2 * 0.5)
        k4 = dt * stream(pos + k3)
        return pos + 1 / 6 * (k1 + k4 + 2 * (k2 + k3))

    rk_4_vectorized = np.vectorize(rk_4, signature="(n)->(n)")

    progression = 0
    print("Integration: ", end="", sep="")
    for step in range(nb_step):
        if int(step / nb_step * 100) - progression >= 5:
            progression += 5
            print(progression, "% ", end="", sep="")
        sl_array[step + 1, :, :] = rk_4_vectorized(sl_array[step, :, :])
    print()

    sl_list = []
    for k in range(nb_sl):
        sl_list.append(
            StreamLine(sl_array[np.isfinite(sl_array[:, k, 0]), k, :], dt)
        )  # ,cut_lines=False))

    return sl_list


def find_eddies(stream_line_list, date=0):
    """Classify stream lines into eddies.

    All the stream lines with a winding angle lower than 2 pi are discarded. If
    the distance between the mean pos of the stream line and an eddy center is
    lower than the mean distance between the mean pos of the stream line and the
    points in the stream line (ie lower that the mean radius, if the stream line
    is considered to be an ellipse), this stream line is added to the eddy. The
    new center of the eddy is computed. Otherwise the stream line is added to a
    new eddy. When all the stream lines have been assigned to an eddy, the
    eddies containing less than 2 (TBD ?) stream lines are discarded.

    Args:
        stream_line_list (list of classes.StreamLine) : The list of trajectories
            to be classified into eddies.
        date (int, default=0) : date at which the trajectories are simulated.

    Returns:
        eddies_list (list of classes.Eddy) : The list of eddies.

    Notes:
        The input file is expected to contain the daily mean fields of east- and
        northward ocean current velocity (uo,vo) in a format as described here:
        http://marine.copernicus.eu/documents/PUM/CMEMS-GLO-PUM-001-024.pdf.

    """

    #  Find a first streamline with a winding angle higher than 2 pi for init
    nb_sl = len(stream_line_list)
    if nb_sl == 0:
        return []
    k0 = 0
    sl = stream_line_list[k0]
    while abs(sl.winding_angle) < 2 * np.pi * 0.9 and k0 < nb_sl - 1:
        k0 += 1
        sl = stream_line_list[k0]
    if nb_sl == k0:
        return []

    sl0 = stream_line_list[k0]
    pre_eddies_list = [[sl0]]
    pre_eddies_center = [sl0.mean_pos]
    pre_eddies_max_radius = [sl0.get_mean_radius()]

    # Put each streamline into an eddy if the winding angle is greater than 2 pi
    for k in range(k0 + 1, len(stream_line_list)):
        sl = stream_line_list[k]
        if abs(sl.winding_angle) < 2 * np.pi * 0.9:
            continue

        #  Mean radius of the stream line
        mean_radius = sl.get_mean_radius()

        # Minimum distance between a stream line center and pre eddies centers
        distances = np.array(pre_eddies_center)
        distances[:, 0] -= sl.mean_pos[0]
        distances[:, 1] -= sl.mean_pos[1]
        distances = np.sqrt(np.sum(distances ** 2, axis=1))
        id_min = np.argmin(distances)
        min_dist = distances[id_min]

        #  If the stream line center is close enougth from a pre eddy center, it
        # is added to the pre eddy.
        if mean_radius > min_dist or pre_eddies_max_radius[id_min] > min_dist:
            pre_eddies_list[id_min].append(sl)
            n = len(pre_eddies_list[id_min])
            sl_center = np.array(
                [pre_eddies_list[id_min][k].mean_pos for k in range(n)]
            )
            sl_nb_pts = np.array(
                [pre_eddies_list[id_min][k].nb_points for k in range(n)]
            )
            sl_wcenter = [sl_center[k] * sl_nb_pts[k] for k in range(n)]
            pre_eddies_center[id_min] = np.sum(sl_wcenter, axis=0) / np.sum(sl_nb_pts)
            if min_dist > pre_eddies_max_radius[id_min]:
                pre_eddies_max_radius[id_min] = min_dist

        # A new eddy is created otherwise
        else:
            pre_eddies_list.append([sl])
            pre_eddies_center.append(sl.mean_pos)
            pre_eddies_max_radius.append(sl.get_mean_radius())

    # Remove pre eddies without enougth stream lines or with a too small radius
    eddies_list = []
    for pre_eddy in pre_eddies_list:
        if len(pre_eddy) >= MIN_STREAMLINE_IN_EDDIES:
            eddy = Eddy(pre_eddy, date)
            if np.sqrt(np.sum(eddy.axis_len ** 2)) > EDDY_MIN_RADIUS:
                eddies_list.append(eddy)

    return eddies_list