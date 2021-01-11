# -*- coding: utf-8 -*-

"""eddy_prediction.py: This file defines the functions used for predict eddies
        with data assimilation methods."""

__author__     = "G. Ghienne, A. Lefebvre, A. Lerosey, L. Menard, A. Perier"
__date__       = "January 2021"
__version__    = "1.0"
__maintainer__ = "Not maintened"
__email__      = ["guillaume.ghienne@imt-atlantique.net",
                  "alexandre.lefebvre@imt-atlantique.net",
                  "antoine.lerosey@imt-atlantique.net",
                  "luc.menard@imt-atlantique.net",
                  "alexandre.perier@imt-atlantique.net"]


from classes import Eddy,Catalog,Observation,ForecastingMethod,FilteringMethod

from AnDA_codes.AnDA_analog_forecasting import AnDA_analog_forecasting
from AnDA_codes.AnDA_data_assimilation import AnDA_data_assimilation

import numpy as np
import pandas as pd



def create_catalog(dt_frame):
    """Compute a catalog from a panda dataframe of previously detected eddies.

    Args:
        dt_frame (pd.DataFrame) : dataframe of eddies containing their dates, id
            and parameters.

    Returns:
        catalog (classes.Catalog) : Catalog containing analogs and successors
            possible to find in the dataframe.

    """

    analogs=np.zeros((1,2))
    successors=np.zeros((1,2))

    # group by id
    for i in dt_frame.id.drop_duplicates():
        df_i=dt_frame[dt_frame.id==i].sort_values('date')

        # find those with successor
        for d in df_i.date:
            if d+1 in df_i.date.values:

                # take their parameters
                analog=df_i[df_i.date==d].to_numpy()[:,2:]
                successor=df_i[df_i.date==d+1].to_numpy()[:,2:]
                analogs=np.concatenate([analogs,analog])
                successors=np.concatenate([successors,successor])

    return Catalog(analogs[1:,:],successors[1:,:])

def calculate_B(catalog):
    """Return a diagonal variance matrix of parametrs variations in the catalog.

    Args:
        catlaog(classes.Catalog) : catalog used later for forecasting.

    Returns:
        B (np.array(6,6)) : diagonal variance matrix of parametrs variations in
            the catalog.

    """
    return np.diag(np.var(catalog.analogs-catalog.successors,axis=0))

def predict_eddy(eddy, catalog, param_weights=np.ones(6),observation=None, R=np.ones((6,6))/36,filtering_method='default', Tmax=10):
    """Compute prediction of the parameters of an eddy.

    Args:
        eddy (classes.Eddy) : the eddy to predict.
        catlaog(classes.Catalog) : catalog to use for forecasting.
        param_weights(np.array(6)) : weights to attribute to each parameter for
            the calcul of closest neighbors/analogs.
        observation(classes.Observation) : Observation classes to optionnaly use
            for data assimilation.
        R (np.array(6,6)) : observation noise covariance matrix.
        filtering_method(classes.FilteringMethod) : filtering paramters class.
        Tmax (int): usefull is obervation=None to know over how much time the
            prediction has to be done.

    Returns:
        prediction : prediction of the eddy parameters in a instance of class similar to observation (attributes values and time).

    """

    # adjustement to attribute desired weights to parameters
    maxis = np.max(np.abs(catalog.analogs),axis=0)
    catalog.analogs=catalog.analogs*param_weights/maxis
    catalog.successors=catalog.successors*param_weights/maxis

    # use default parameters
    if filtering_method=='default':

        B = calculate_B(catalog)
        forecasting_method = ForecastingMethod(catalog)
        filtering_method = FilteringMethod(B,R,forecasting_method)

    # adjust B to weights
    filtering_method.B=filtering_method.B*param_weights/maxis

    # find initial eddy parameter
    [x,y] = eddy.center
    [L,l] = eddy.axis_len
    d = eddy.axis_dir[0]
    w = eddy.angular_velocity

    # conclude the set up of the filtering class giving it the first eddy
    filtering_method.set_first_eddy(np.array([x,y,L,l,d,w])*param_weights/maxis)

    # create a nan observation is observation=None
    if observation==None:
        values=np.full((Tmax,6), np.nan)
        time=np.arange(Tmax)
        observation=Observation(values,time)

    # adjust observation with corresponding weights
    observation.values=observation.values*param_weights/maxis

    # compute data assimilation
    prediction=AnDA_data_assimilation(observation, filtering_method)

    # inverse adjustement of weights
    prediction.values=prediction.values*maxis/param_weights

    return(prediction)


def get_eddy(prediction,t):
    """Return a classes.Eddy instance corresponding to time t of predictions

    Args:
        prediction : prediction compute by data assimilation

    Returns:
        eddy (classes.Eddy) : eddy corresponding to demanded time.

    """
    return Eddy([],prediction.values[t,:])
