import numpy as np
from numba import njit
from tqdm import tqdm

#######################################################
##                  SHPERICAL
@njit
def trialWaveSpherical(r, n_particles,n_dimensions,alpha,beta,hardCoreDiameter=0):
        wave = 0
        for i in range(n_particles):
            square = 0 # r**2
            for d in range(n_dimensions):
                square += r[i,d]**2
            wave -= alpha*square
        return np.exp(wave)

@njit 
def dPsiSpherical(r, n_particles,n_dimensions,alpha,beta):
    derivative = 0
    for i in range(n_particles):
        square = 0
        for d in range(n_dimensions):
            square += r[i,d]**2
        derivative -= square
    return derivative

#######################################################
##                  ELLIPTICAL
@njit
def trialWaveElliptical(r, n_particles,n_dimensions,alpha,beta,hardCoreDiameter=.00433):
    single = 0
    correlation = 1
    for i in range(n_particles):
        square = 0 # r**2
        for d in range(n_dimensions):
            if d == 2:
                square += beta*(r[i,d])**2
            else:
                square += r[i,d]**2
        single -= alpha*square 
        for j in range(i+1,n_particles):
            r_ij = 0
            for d in range(n_dimensions):
                r_ij += (r[i,d] - r[j,d])**2
            r_ij = np.sqrt(r_ij)
            if r_ij < hardCoreDiameter:
                correlation *= 0
            else:
                correlation *= 1 - hardCoreDiameter/(np.sqrt(r_ij)) # Correlation
                
    return np.exp(single)*correlation

@njit(parallel = True, nogil = True)
def dPsiElliptical(r, n_particles,n_dimensions,alpha,beta,hardCoreDiameter=.00433):
    derivative = 0
    for i in range(n_particles):
        for d in range(n_dimensions):
            if d == 2:
                derivative += beta*(r[i,d])**2
            else:
                derivative += r[i,d]**2
    return -derivative