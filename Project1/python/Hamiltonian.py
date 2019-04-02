import numpy as np
from numba import njit
from tqdm import tqdm
from GradientFunctions import *
#######################################################
##                  SHPERICAL
@njit
def HamiltonianSpherical(r, n_particles,n_dimensions,alpha,beta,hardCoreDiameter=0):
    oneBody = alpha*n_dimensions*n_particles
    square = 0
    for i in range(n_particles):
        for d in range(n_dimensions):
            square += r[i,d]**2
    oneBody += (0.5 - 2*alpha**2)*square
    return oneBody

#######################################################
##                  ELLIPTICAL

@njit
def innerProduct(a,b):
    if a.shape[0] != b.shape[0]:
        raise ValueError('Vector must be of equal size!')
    prod = 0
    for d in range(a.shape[0]):
        prod += a[d]*b[d]
    return prod

@njit
def HamiltonianElliptical(r, n_particles,n_dimensions,alpha,beta,hardCoreDiameter=.00433):
    """
    Expectation value for the energy of the system.
    """
    P = 0
    K = 0
    xum = 0
    for i in range(n_particles):
        # Kinetic Energy
        r_i = r[i]
        laplacian  = laplaceSingleParticle(r_i, n_dimensions,alpha,beta)
        gradientSP  = gradientSingleParticle(r_i, n_dimensions,alpha,beta)
        gradientC,gradientC2   = gradientCorrelation(r,r_i,i, n_particles,n_dimensions,hardCoreDiameter)
        for d in range(n_dimensions):
            laplacian += 2*gradientSP[d]*gradientC[d]
        laplacian += laplaceCorrelation(r,r_i,i, n_particles,n_dimensions,hardCoreDiameter) + gradientC2

        # Potential Energy
        square = 0 # r**2
        for d in range(n_dimensions):
            if d == 2:
                square += (beta*r_i[d])**2
            else:
                square += r_i[d]**2
        
        K += laplacian
        P += square

    return 0.5*(P - K)