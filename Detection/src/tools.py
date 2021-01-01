# -*- coding: utf-8 -*-

"""classes.py: This file defines some functions used for eddies detection and
representation."""

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
import matplotlib.pyplot as plt

EDDY_MIN_AXIS_LEN =1e-2
EDDY_AXIS_PRECISION = 1e-4


def grad_desc(f_error,f_grad):
    d0= 1
    a,b = 1,1
    epsilon = EDDY_AXIS_PRECISION+1
    new_err = f_error(a,b)
    while epsilon > EDDY_AXIS_PRECISION:
        da,db = d0,d0
        
        err_a = new_err
        grad_a = np.sign(f_grad(a,b)[0])
        while f_error(a-grad_a*da,b)>err_a or a-grad_a*da<EDDY_MIN_AXIS_LEN:
            da /= 2.
        a -= grad_a*da
        
        err_b = f_error(a,b)
        grad_b = np.sign(f_grad(a,b)[1])
        while f_error(a,b-grad_b*db)>err_b or b-grad_b*db<EDDY_MIN_AXIS_LEN:
            db /= 2.
        b -= grad_b*db
        
        new_err = f_error(a,b)
        epsilon = np.sqrt(da*da+db*db)

    return a,b