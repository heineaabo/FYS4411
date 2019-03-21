import numpy  as     np
from   random import random,seed,normalvariate
from   math   import exp,sqrt
from pandas import DataFrame
from   time   import perf_counter as clock
from tqdm import tqdm
import sys
from tools import *
from numba import jit

# BASH: python fastSpherical.py {n_runs} {n_particles} {n_dimesions} {alpha_indx}
#   EX: python fastSpherical.py 2**20 10 3 4

### HAMILTONIAN
@jit(nopython=True)
def localEnergy(r,n_particles,n_dimensions,alpha):
    oneBody = alpha*n_dimensions*n_particles
    C = (0.5 - 2*alpha**2)
    for i in range(n_particles):
        square = 0
        for d in range(n_dimensions):
            square += r[i,d]**2
        oneBody += C*square
    return oneBody 

### WAVE FUNCTION
@jit(nopython=True)
def trialWave(r,n_particles,n_dimensions,alpha):
    wave = 0
    for i in range(n_particles):
        square = 0
        for d in range(n_dimensions):
            square += r[i,d]**2
        wave -= alpha*square
    return exp(wave)  

### QUANTUM FORCE
@jit(nopython=True)
def quantumForce(r,alpha):
    return -4*alpha*r

### Gradient
#@jit(nopython=True)
def gradient(r_k,n_dimensions,vector=False):
    """
    Gradient of single-particle function.
    Returns vector with elements for each dimension.
    """
    gradient = np.zeros(n_dimensions)
    for d in range(n_dimensions):
        gradient[d] += r_k[d]
    if vector:
        return -2*alpha*gradient
    else:
        return -2*alpha*np.sum(gradient)

### Functions
@jit(nopython=True)
def monteRun(n_particles,n_dimensions,alpha,beta,
            posOld,posNew,trialNew,trialOld,qOld,qNew,
            dE,
            accepted,positive,negative,importanceSampling=True):
    for i in range(n_particles):
        if importanceSampling:
            greenOld = .0
            greenNew = .0
        for j in range(n_dimensions):
            posNew[i,j] = posOld[i,j] + sqrt(stepSize)*np.random.normal(0.0,1.0) + D*stepSize*qOld[i,j]

        trialNew = trialWave(posNew,n_particles,n_dimensions,alpha)
        qNew     = quantumForce(posNew,alpha)
        
        # Metropolis-Hasting
        if importanceSampling:
            for j in range(n_dimensions):
                greenOld -= (posNew[i,j] - posOld[i,j] -D*stepSize*qOld[i,j])**2
                greenNew -= (posOld[i,j] - posNew[i,j] -D*stepSize*qNew[i,j])**2
            Greens = exp(greenOld)/exp(greenNew)

            transRatio = Greens * trialNew**2 / trialOld**2
                
            if np.random.normal(0.0,1.0) < transRatio: #random()
                accepted += 1
                for j in range(n_dimensions):
                    posOld[i,j] = posNew[i,j]
                    qOld[i,j] = qNew[i,j]
                trialOld = trialNew
                dE       = localEnergy(posOld,n_particles,n_dimensions,alpha)
                if dE >= 0:
                    positive += 1
                if dE < 0:
                    negative += 1
        
        # Brute Force
        else:
            transRatio = trialNew**2 / trialOld**2
            if np.random.uniform(0.0,1.0) < transRatio:
                accepted += 1
                for j in range(n_dimensions):
                    posOld[i,j] = posNew[i,j]
                    qOld[i,j] = qNew[i,j]
                trialOld = trialNew
                dE       = localEnergy(posOld,n_particles,n_dimensions,alpha)
                if dE >= 0:
                    positive += 1
                if dE < 0:
                    negative += 1
    return dE,accepted,positive,negative


if __name__ == '__main__':
    # System variables
    n_particles = int(sys.argv[2])
    n_dimensions = int(sys.argv[3])
    # Simulation variables
    n_runs   = eval(sys.argv[1])
    alphs = np.array([0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.8])
    alpha   = alphs[int(sys.argv[4])]
    beta    = 1
    stepSize = 0.005
    D = 0.5
    # Simulation matrices
    posOld   = np.zeros((n_particles,n_dimensions))
    posNew   = np.zeros((n_particles,n_dimensions))
    trialNew = 0
    trialOld = 0
    qNew     = np.zeros((n_particles,n_dimensions))
    qOld     = np.zeros((n_particles,n_dimensions))
    # Energies and statistics
    E  = 0
    E2 = 0
    dE = 0
    MC_energies = np.zeros(n_runs)

    # INIT POSOLD
    for i in range(n_particles):
        for j in range(n_dimensions):
            posOld[i,j] = np.sqrt(stepSize) * (random() - .5)
        #for j in range(i):
        #    if relativeDistance(position[i],position[j]) < hardCoreDiameter:
        #        print('Wrong')
    qOld = quantumForce(posOld,alpha)
    trialOld = trialWave(posOld,n_particles,n_dimensions,alpha)
    # MONTECARLO
    accepted = 0
    positive = 0
    negative = 0
    for n in tqdm(range(n_runs)):
        dE,accepted,positive,negative = monteRun(n_particles,n_dimensions,alpha,beta,
            posOld,posNew,trialNew,trialOld,qOld,qNew,
            dE,
            accepted,positive,negative,importanceSampling=True)
        E  += dE
        E2 += dE**2
        MC_energies[n] = dE
    E  /= n_runs
    E2 /= n_runs
    var = E2 - E**2

    print('Accepted:',accepted)
    print('Positive:{}, Negative: {}'.format(positive,negative))

    ENERGY = E                    # Energy
    VARIANCE = var,               # Variance
    ERROR = np.sqrt(var/n_runs)   # Error
    df = DataFrame(ENERGY,index = np.array([alpha]),columns=['Energy'])
    df['Variance'] = VARIANCE
    df['Error'] = ERROR
    print(df)
