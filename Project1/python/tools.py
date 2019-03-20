import numpy as np

def relativeDistance(r_k,r_l):
    dist = 0
    for d in range(r_k.shape[0]):
        dist += (r_k[d] - r_l[d])**2
    return np.sqrt(dist)

def simpleDistance(r_k,r_l):
    dist = np.zeros_like(r_k)
    for d in range(r_k.shape[0]):
        dist[d] += (r_k[d] - r_l[d])
    return dist

def innerProduct(r_k,r_l):
    prod = 0
    for i in range(r_k.shape[0]):
        prod += r_k[i]*r_l[i]
    return prod