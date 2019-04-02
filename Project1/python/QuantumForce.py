import numpy as np
from numba import njit
from tqdm import tqdm
from GradientFunctions import *

#######################################################
##                  SHPERICAL
@njit
def quantumForceSpherical(r, n_particles,n_dimensions,alpha,beta,hardCoreDiameter=0):
    return -4*alpha*r

#######################################################
##                  ELLIPTICAL
@njit
def quantumForceElliptical(r, n_particles,n_dimensions,alpha,beta,hardCoreDiameter=.00433):
        """
        Quantum force of the system
        Returns matrix.
        """
        qForce = np.zeros((n_particles,n_dimensions))
        vector = np.ones(n_dimensions)
        for k in range(n_particles):
            gradC,g2 = gradientCorrelation(r,r[k],k,n_particles,n_dimensions,hardCoreDiameter)
            qForce[k] = 2 * (gradientSingleParticle(r[k],n_dimensions,alpha,beta) + gradC)
        return qForce
