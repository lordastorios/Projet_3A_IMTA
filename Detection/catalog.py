# -*- coding: utf-8 -*-

"""classes.py: This file defines the functions used for writting and reading
the caltalog as a csv file."""

__author__     = "G. Ghienne, A. Lefebvre, A. Lerosey, L. Menard, A. Perier"
__date__       = "December 2020"
__version__    = "1.0"
__maintainer__ = "Not maintened"
__email__      = ["guillaume.ghienne@imt-atlantique.net",
                  "alexandre.lefebvre@imt-atlantique.net",
                  "antoine.lerosey@imt-atlantique.net",
                  "luc.menard@imt-atlantique.net",
                  "alexandre.perier@imt-atlantique.net"]


import numpy as np
import pandas as pd


def write_catalog(data,fname="./catalog.csv"):
    """ Convert observations and tracking of eddies into a csv file.

    Observations and tracking of eddies are expected to be of the same format as
    the results of the function 'eddies_tracking.eddies_tracker'.

    Args:
        fname (string, default = "./catalog.csv") : The file name in which the
            catalog is saved.
        data (dict(int : dict(int : classes.Eddy))) : The path of eddies.
            A unique id is assigned to each eddy so that 2 object 'classes.Eddy'
            share the same id if and only if they represent the same eddy at
            different dates. The first key is a date, the second key is the eddy
            identifier.

    Returns:
        df (pandas.Dataframe) : The data frame as it as been saved.

    """

    # Organise the data in a list.
    rows = []
    for date,eddies in data.items():
        for eddy_id,eddy in eddies.items():
            angle = np.angle(eddy.axis_dir[0,0]+eddy.axis_dir[1,0]*1j)
            rows.append([date,eddy_id,eddy.center[0],eddy.center[1],
                        eddy.axis_len[0],eddy.axis_len[1],angle,
                        eddy.angular_velocity])
            
    # Same the dataframe as a csv file.
    df = pd.DataFrame(rows,index=None,columns=['date','id','center_x','center_y','a','b','angle','omega'])
    df.to_csv(fname)

    return df