# ------------------------- IMPORTS -------------------------

import cv2
import time
import numpy as np
import math
import os
import sys

from Thymio import Thymio




# -------------------- GLOBAL VARIABLES ---------------------

# Inizalizing variables for Kalman filter

# We assume the system to respect a constant velocity model where the velocity in cells/degrees per motion
# we use kalman to reestimate the position and a possible correction factor for the speed

# To filter using the camera detection data, we project the system with a C matrix that is an identity for the three first values (x,y, theta)
C = np.concatenate((np.identity(3), np.zeros([3,3])), axis = 1)

# After tuning we define the variances values as following
# obiouvsly, the camera is supposed to be more trustful than the imprecise odometry

Q               = np.diag([10, 10, 10, 0.001, 0.001, 0.001])  # stateserror
R               = np.diag([1, 1, 1])                          # measurement error


# ------------------------ FUNCTIONS ------------------------



# The A matrix is recalculated for every Kalman iteration to match the assumption of a constant velocity system

def accumulateAMatrix(A,motion):
    A[0][3] += motion[0]
    A[1][4] += motion[1]
    A[2][5] += motion[2]
    return A
    
def resetAMatrix():
    A = np.identity(6)
    return A


# Filter main function

def kalman(y, A, x_prec, SigmaOld):
    x_prior     = A@x_prec
    Sigma_est   = A@SigmaOld@A.T + Q
    e           = y - C@x_prior
    S           = C@Sigma_est@C.T + R
    K           = Sigma_est@C.T@np.linalg.inv(S)
    x_filtered  = x_prior + K@e
    Sigma       = (np.identity(6) - K@C)@Sigma_est
    SigmaOld    = Sigma
    x_prec      = x_filtered
    filteredPos = x_filtered[0:3]
    return filteredPos, x_prec, SigmaOld, x_prior[0:3]
