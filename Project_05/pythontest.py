#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script demonstrates how to compute the given resonant toroidal mode 
Period times (in seconds).
"""

# Import the functions necessary for the computation
# This also includes the particular modes of interest

import TF
import numpy as np

# Set an initial decision variable vector

x=[ 0.6 ,  2.6 ,  -3.6 ,  7.0 , -7.0 , 11.2 , -1.6 ,  5.0 ,
   -3.0 ,  5.6 ,  -6.4 ,  8.0 ,  5.6 , -1.0 , -4.4 ,  8.8 ,
  -18.6 , 22.2 ,  -4.8 , 10.0 ,  0.8 , -2.0 ,-17.2 , 22.4 ,
   -9.2 , 17.2 , -14.0 , 11.4 ,  1.0 , -2.2 ,  1.4 ,  6.4   ]

# call the ToroidalPeriod calculator
# Tc is an array of computed periods.
# Te is an array of corresponding experimental periods

Tc,Te = TF.ToroidalPeriods(np.array(x))

# Compute an objective function (this is an example).
if np.isnan(Tc[0]):
    f=np.inf
else:
    f=np.linalg.norm(Tc-Te)/np.linalg.norm(Te)
    
print('Objective Value = ',f)


# Objective Value =  0.004888379794193573


