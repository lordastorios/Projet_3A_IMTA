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

from constants import EDDY_MIN_AXIS_LEN, EDDY_AXIS_PRECISION


def grad_desc(f_error,f_grad):
    """ Compute a local minimum of a function using gradient descente.
    
    Compute the optimal axis length of the ellipse representing an eddy, using
    an error function and its gradient.

    Args:
        f_error (function(a: float, b: float) = error: float) : 
        f_grad (function(a: float, b: float) = gradient: np) : 

    Returns:
        a,b (float,float) : The optimal ellipse axis length.

    """

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


def pts_is_in_ellipse(pts_list,a,b,angle,center):
    """ Test if a point is inside an ellipse.

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
    centered_pts[:,0] -= center[0]
    centered_pts[:,1] -= center[1]

    rotation = np.array([[np.cos(angle),np.sin(angle)],
                        [-np.sin(angle),np.cos(angle)]])
    aligned_pts = np.dot(rotation,centered_pts.T)

    return (aligned_pts[0,:]**2/a**2 + aligned_pts[1,:]**2/b**2)<=1