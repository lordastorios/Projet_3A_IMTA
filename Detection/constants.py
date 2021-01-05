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


EDDY_MIN_AXIS_LEN =1e-2        # Axis length for ellipses representing eddies.
EDDY_AXIS_PRECISION = 1e-4     # Stop criterion for gradient descente.
EQUATORIAL_EARTH_RADIUS = 6378.137e3
MIN_STREAMLINE_IN_EDDIES = 3   # Number for streamlines in eddies.