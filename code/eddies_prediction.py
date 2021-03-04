# -*- coding: utf-8 -*-

"""eddy_prediction.py: This file defines the functions used for predict eddies
        with data assimilation methods."""

__author__ = "G. Ghienne, A. Lefebvre, A. Lerosey, L. Menard, A. Perier"
__date__ = "January 2021"
__version__ = "1.0"
__maintainer__ = "Not maintened"
__email__ = [
    "guillaume.ghienne@imt-atlantique.net",
    "alexandre.lefebvre@imt-atlantique.net",
    "antoine.lerosey@imt-atlantique.net",
    "luc.menard@imt-atlantique.net",
    "alexandre.perier@imt-atlantique.net",
]


from classes import (
    Eddy,
    Catalog,
    Observation,
    ForecastingMethod,
    FilteringMethod,
)

from AnDA.AnDA_data_assimilation import AnDA_data_assimilation

import numpy as np


def create_catalog(dt_frame):
    """Compute a catalog from a panda dataframe of previously detected eddies.

    Args:
        dt_frame (pd.DataFrame) : dataframe of eddies containing their dates,
            id and parameters.

    Returns:
        catalog (classes.Catalog) : Catalog containing analogs and successors
            possible to find in the dataframe.

    """

    analogs = np.zeros((1, 6))
    successors = np.zeros((1, 6))

    # group by id
    for i in dt_frame.id.drop_duplicates():
        df_i = dt_frame[dt_frame.id == i].sort_values("date")

        # find those with successor
        for d in df_i.date:
            if d + 1 in df_i.date.values:

                # take their parameters
                analog = df_i[df_i.date == d].to_numpy()[:, 2:]
                successor = df_i[df_i.date == d + 1].to_numpy()[:, 2:]
                analogs = np.concatenate([analogs, analog])
                successors = np.concatenate([successors, successor])

    return Catalog(analogs[1:, :], successors[1:, :])


def predict_eddy(eddy, catalog, observation, k=20, N=50, search="complet"):
    """Compute prediction of the parameters of an eddy.

    Args:
        eddy (classes.Eddy) : the eddy to predict.
        catlaog(classes.Catalog) : catalog to use for forecasting.
        observation(classes.Observation) : Observation classes to optionnaly
            use for data assimilation.
        k (int) : number of analogs to use during the forecast.
        N (int) : number of members.
        search (string) : "complet" or "fast" to use either Jaccard Index
            based or just Euclidian based metric.

    Returns:
        prediction : prediction of the eddy parameters in a instance of class
            similar to observation (attributes values and time).

    """

    # create a nan observation is observation=None
    if type(observation) == int:
        values = np.full((observation, 6), np.nan)
        time = np.arange(observation)
        observation = Observation(values, time)

    # set up forecasting and filtering classes
    forecasting_method = ForecastingMethod(catalog, k, search=search)
    filtering_method = FilteringMethod(observation.R, N, forecasting_method)

    # find initial eddy parameter
    [x, y] = eddy.center
    [L, l] = eddy.axis_len
    d = np.angle(eddy.axis_dir[0, 0] + eddy.axis_dir[1, 0] * 1j)
    w = eddy.angular_velocity

    # conclude the set up of the filtering class giving it the first eddy
    filtering_method.set_first_eddy(np.array([x, y, L, l, d, w]))

    # compute data assimilation
    prediction = AnDA_data_assimilation(observation, filtering_method)

    return prediction


def predict_eddies(
    catalog, observation, R, k=20, N=50, t_vanish=10, search="fast", Tmax=None
):
    """Compute prediction of the parameters of an eddy.

    Args:
        catalog(pandas.DataFram) : catalog to use for forecasting.
        observation(pandas.DataFram) : Observation to use for AnDA.
        R (np.array(6,6)) : observation noise covariance matrix.
        k (int) : number of analogs to use during the forecast.
        N (int) : number of members.
        t_vanish (int) : time after wich (and before witch) we consider an
            unobserved eddy has vanished.
        search (string) : "complet" or "fast" to use either Jaccard Index
            based or just Euclidian based metric.
        Tmax (int): time during which we want predictions (max time of
            observation.date if None).

    Returns:
        eddies (dict[time][list(eddy)]) : list of eddies for each used time.

    """

    eddies = {}
    prediction = []
    catalog = create_catalog(catalog)

    date = observation.date.drop_duplicates().to_numpy()
    if Tmax is None:
        date = range(date[0], date[-1] + 1)
    else:
        date = range(date[0], date[0] + Tmax)

    if t_vanish is None:
        t_vanish = np.inf

    for id in observation.id.drop_duplicates():
        obs = create_observation(observation, R, id, Tmax)
        if obs.values[~np.isnan(obs.values)].shape[0] != 0:
            if np.isnan(obs.values[0, 0]):
                first_obs = np.where(~np.isnan(obs.values[:, 0]))[0][-1]
                if first_obs >= t_vanish:
                    obs.values = obs.values[first_obs - t_vanish :, :]
                    obs.time = obs.time[first_obs - t_vanish :]
            else:
                first_obs = 0
            if np.isnan(obs.values[-1, 0]):
                last_obs = np.where(~np.isnan(obs.values[:, 0]))[0][-1]
                if first_obs + last_obs <= date[-1] - t_vanish:
                    obs.values = obs.values[: last_obs + t_vanish, :]
                    obs.time = obs.time[: last_obs + t_vanish]
            param = obs.values[~np.isnan(obs.values)[:, 0], :][0, :]
            eddy = Eddy([], 0, param=param)
            prediction.append(predict_eddy(eddy, catalog, obs, k, N, search))

    for t in date:
        eddies[t] = []
        for pred in prediction:
            tmin = pred.time[0]
            tmax = pred.time[-1]
            if t >= tmin and t <= tmax:
                eddies[t].append(get_eddy(pred, t - tmin))

    return eddies


def get_eddy(prediction, t):
    """Return a classes.Eddy instance corresponding to time t of predictions

    Args:
        prediction : prediction compute by data assimilation

    Returns:
        eddy (classes.Eddy) : eddy corresponding to demanded time.

    """
    return Eddy([], t, param=prediction.values[t, :])


def create_observation(dt_frame, R=np.eye(6) / 36, id=None, Tmax=None):
    """Return a classes.Observation instance corresponding to the observations
    made of a given eddy

    Args:
        dt_frame (pd.DataFrame) : a data frame of differents observed eddies
            containing date,id and parameters.
        R (np.array(6,6)) : observation noise covariance matrix.
        id (int) : id of the eddy on which data assimilation will be done.
        Tmax (int): time during which we want predictions (max time of
            observation.date if None).

    Returns:
        observation (classes.Observation) : corresponding instance.

    """
    if id is None:
        id = dt_frame.id[0]
    df_id = dt_frame[dt_frame.id == id]

    tmin, tmax = dt_frame.date.min(), dt_frame.date.max()
    if Tmax is None:
        T = tmax - tmin + 1
    else:
        T = Tmax

    time = np.arange(tmin, tmin + T)

    values = np.zeros((T, 6))

    for t in range(T):
        if t + tmin in df_id.date.values:
            values[t, :] = df_id[df_id.date == t + tmin].to_numpy()[:, 2:]
        else:
            values[t, :] = np.nan

    return Observation(values, time, R)
