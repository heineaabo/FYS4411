import numpy as np
from numba import njit
from tqdm import tqdm

@njit
def gradientSingleParticle(r_k, n_dimensions,alpha,beta):
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

@njit 
def laplaceSingleParticle(r_k, n_dimensions,alpha,beta):
    """
    Laplacian of the single-particle function.
    Returns scalar.
    """
    gradient = 0 # r**2
    for d in range(n_dimensions):
        if d == 2:
            gradient += (beta*r_k[d])**2
        else:
            gradient += r_k[d]**2
    return -2*alpha*(2 + beta) + 4*(alpha**2)*gradient 

@njit
def gradientCorrelation(r,r_k,k, n_particles,n_dimensions,hardCoreDiameter=.00433):
    """
    Gradient of correlation function.
    Returns vector
    """
    gradient = np.zeros(n_dimensions)
    gradient2 = 0
    for l in range(n_particles):
        if l == k:
            continue
        r_l = r[l]

        relDist = np.zeros(n_dimensions) # Vector distance
        for d in range(n_dimensions):
            relDist[d] = (r_k[d]-r_l[d])

        eucDist = 0 # Euclidean distance
        for d in range(n_dimensions):
            eucDist += (r_k[d]-r_l[d])**2

        r_kl = np.sqrt(eucDist)
        derivative = hardCoreDiameter/(r_kl*(r_kl-hardCoreDiameter))
        for d in range(n_dimensions):
            gradient[d] += (relDist[d]/r_kl)*derivative
            gradient2 += ((relDist[d]/r_kl)*derivative)**2
    return gradient,gradient2#*(derivative/r_kl)

@njit    
def laplaceCorrelation(r,r_k,k, n_particles,n_dimensions,a=.00433):
    """
    Laplacian of correlation function.
    Returns scalar.
    """
    laplace = 0
    for l in range(n_particles):
        if l == k:
            continue
        r_l = r[l]
        eucDist = 0 # Euclidean distance
        for d in range(n_dimensions):
            eucDist += (r_k[d]-r_l[d])**2
        r_kl = np.sqrt(eucDist)

        #laplace += (2 * a)/((r_kl**2)*(r_kl-a))
        #laplace += (a**2 - 2*a*r_kl)/((r_kl**2)*(r_kl-a)**2)
        laplace += (-a**2)/((r_kl**2)*(r_kl-a)**2)

    return laplace