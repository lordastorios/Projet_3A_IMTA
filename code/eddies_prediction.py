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

    analogs=np.zeros((1,6))
    successors=np.zeros((1,6))

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

def predict_eddy(eddy, catalog,observation=None,filtering_method='default', Tmax=10,param_weights=np.ones(6),R=np.ones((6,6))/36):
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

    # create a nan observation is observation=None
    if observation==None:
        values=np.full((Tmax,6), np.nan)
        time=np.arange(Tmax)
        observation=Observation(values,time)

    # adjustement to attribute desired weights to parameters
    multi = param_weights/np.max(np.abs(catalog.analogs),axis=0)
    catalog.adjust(multi)
    observation.adjust(multi)

    # use default parameters
    if filtering_method=='default':
        forecasting_method = ForecastingMethod(catalog)
        filtering_method = FilteringMethod(R*multi**2,forecasting_method)

    else:
        # adjust B and R to weights
        filtering_method.adjust(multi**2)

    # find initial eddy parameter
    [x,y] = eddy.center
    [L,l] = eddy.axis_len
    d = np.angle(eddy.axis_dir[0,0]+eddy.axis_dir[1,0]*1j)
    w = eddy.angular_velocity

    # conclude the set up of the filtering class giving it the first eddy
    filtering_method.set_first_eddy(np.array([x,y,L,l,d,w])*multi)

    # compute data assimilation
    prediction=AnDA_data_assimilation(observation, filtering_method)

    # inverse adjustement of weights
    prediction.values=prediction.values/multi
    observation.adjust(1/multi)

    return(prediction)


def get_eddy(prediction,t):
    """Return a classes.Eddy instance corresponding to time t of predictions

    Args:
        prediction : prediction compute by data assimilation

    Returns:
        eddy (classes.Eddy) : eddy corresponding to demanded time.

    """
    return Eddy([],param=prediction.values[t,:])

def create_observation(dt_frame,id):
    """Return a classes.Observation instance corresponding to the observations
    made of a given eddy

    Args:
        dt_frame (pd.DataFrame) : a data frame of differents observed eddies
            containing date,id and parameters.
        id (int) : id of the eddy on which data assimilation will be done.

    Returns:
        observation (classes.Observation) : corresponding instance.

    """
    df_id=dt_frame[dt_frame.id==id]

    tmin,tmax=dt_frame.date.min(),dt_frame.date.max()
    T=tmax-tmin+1
    time=np.arange(tmin,tmax+1)

    values=np.zeros((T,6))

    for t in range(T):
        if t+tmin in df_id.date.values:
            values[t,:]=df_id[df_id.date==t+tmin].to_numpy()[:,2:]
        else:
            values[t,:]=np.nan

    return Observation(values,time)
