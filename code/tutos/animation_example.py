import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("../")

import numpy as np
import pandas as pd

from eddies_prediction import predict_eddies
from plot import Animation


### initialization of catalog and observations

catalog = pd.read_csv("catalog.csv", usecols=list(range(1, 9)))
observation = pd.read_csv("catalog300.csv", usecols=list(range(1, 9)))


### data assimilation

R = np.eye(6) / 1000  # covariance of observation noise
prediction = predict_eddies(catalog, observation, R, t_vanish=None, Tmax=30)


### animation

animation = Animation(prediction)
animation.show()