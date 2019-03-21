import numpy as np
from tools import *
from numba import jit

### Functions  
@jit(nopython=True)
def gradientSingleParticle(r_k,n_dimensions,alpha,beta):
    """
    Gradient of single-particle function.
    Returns vector with elements for each dimension.
    """
    gradient = np.zeros(n_dimensions)
    for d in range(n_dimensions):
        if d == 2:
            gradient[d] += beta*r_k[d]
        else:
            gradient[d] += r_k[d]
    return -2*alpha*gradient

@jit(nopython=True)
def laplaceSingleParticle(n_particles,n_dimensions,alpha,beta,r_k):
    """
    Laplacian of the single-particle function.
    Returns scalar.
    """
    gradient = 0
    for d in range(n_dimensions):
        if d == 2:
            gradient += (beta*r_k[d])**2
        else:
            gradient += r_k[d]**2
    return -2*alpha*(2 + beta) + 4*(alpha**2)*gradient 

@jit(nopython=True)
def gradientCorrelation(r,n_particles,n_dimensions,beta,r_k,k,a):
        """
        Gradient of correlation function.
        Returns vector
        """
        gradient = np.zeros(n_dimensions)
        for l in range(n_particles):
            if l == k:
                continue
            r_l=r[l]
            dist = np.zeros(n_dimensions)
            for d in range(n_dimensions):
                if d == 2:
                    dist[d] = (beta*r_k[d] - beta*r_l[d])
                else:
                    dist[d] = (r_k[d] - r_l[d])
            nominator = dist
            dist2 = 0
            for d in range(n_dimensions):
                if d == 2:
                    dist2 += (beta*r_k[d] - beta*r_l[d])**2
                else:
                    dist2 += (r_k[d] - r_l[d])**2
            r_kl = np.sqrt(dist2)
            derivative = a/(r_kl*(r_kl-a))
            for d in range(n_dimensions):
                gradient[d] = (nominator[d]/r_kl)*derivative
        return gradient

@jit(nopython=True)
def laplaceCorrelation(r,n_particles,n_dimensions,beta,r_k,k,a):
    """
    Laplacian of correlation function.
    Returns scalar.
    """
    laplace = 0
    for l in range(n_particles):
        if l == k:
            continue
        r_l = r[l]
        euclideanDistance = 0
        for d in range(n_dimensions):
            if d == 2:
                euclideanDistance += (beta*r_k[d]-beta*r_l[d])**2
            else:
                euclideanDistance += (r_k[d]-r_l[d])**2
        r_kl = np.sqrt(euclideanDistance)
        laplace += (2/r_kl)*(a/(r_kl*(r_kl-a))) + (a**2 - 2*a*r_kl)/((r_kl**2)*(r_kl-a)**2)
    return laplace

@jit(nopython=True)
def distancesMatrix(r,n_particles,n_dimensions):
    distances = np.zeros((n_particles,n_particles))
    for i in range(n_particles):
        for j in range(i+1,n_particles):
            dist = 0
            for d in range(n_dimensions):
                if d==2:
                    dist += (beta*r[i,d] - beta*r[j,d])**2
                else:
                    dist += (r[i,d] - r[j,d])**2
            distances[i,j] = np.sqrt(dist)
    return distances
