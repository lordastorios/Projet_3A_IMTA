# -*- coding: utf-8 -*-

"""classes.py: This file gathers the constants and global parameters for the
detection."""

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

from tools import pts_is_in_ellipse
from constants import JACCARD_RESOLUTION


def eddies_jaccard_index(e1,e2):
    """ Compute the Jaccard index of 2 eddies represented by ellipses.

    The ellipse parameters are the center of the eddy and the axis length and
    direction. The suface of the intersection is computed using finit elements.

    Args:
       e1, e2 (classes.Eddy) : The eddies for which the Jaccard index in computed.

    Returns:
        index (float) : The Jaccard index of the 2 ellipses.

    """

    max_r1 = max(e1.axis_len)
    max_r2 = max(e2.axis_len)

    if max_r1>max_r2:
        return eddies_jaccard_index(e2,e1)

    # Ellipse 1 parameters.
    a1,b1      = e1.axis_len
    surface_e1 = a1*b1*np.pi
    angle1     = np.angle(e1.axis_dir[0,0]+1j*e1.axis_dir[1,0])

    # Ellipse 2 parameters.
    a2,b2      = e2.axis_len
    surface_e2 = a2*b2*np.pi
    angle2     = np.angle(e2.axis_dir[0,0]+1j*e2.axis_dir[1,0])

    # Generate a list of points for computing the intersecton surface with
    # finit elements.
    nb_points = JACCARD_RESOLUTION
    x = np.linspace(e1.center[0]-max_r1,e1.center[0]+max_r1,nb_points)
    y = np.linspace(e1.center[1]-max_r1,e1.center[1]+max_r1,nb_points)
    xx,yy = np.meshgrid(x,y)
    points_list = np.array(list(zip(xx.flatten(),yy.flatten())))

    # Surface represented by a points.
    ds = (2*max_r1/nb_points)**2

    # Count the number of points in the intersection.
    inside_e1 = pts_is_in_ellipse(points_list,a1,b1,angle1,e1.center)
    points_in_e1 = points_list[inside_e1]
    inside_e1_and_e1 = pts_is_in_ellipse(points_in_e1,a2,b2,angle2,e2.center)
    surface_e1_and_e2 = ds*np.sum(inside_e1_and_e1)

    index = surface_e1_and_e2/(surface_e1 + surface_e2 - surface_e1_and_e2)

    return index