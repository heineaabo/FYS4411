import numpy  as     np
from   random import random,seed,normalvariate
from   math   import exp,sqrt
from pandas import DataFrame
from   time   import perf_counter as clock
from tqdm import tqdm
import sys
from tools import *
from functionsElliptical import *
from numba import jit

# BASH: python fastElliptical.py {n_runs} {n_particles} {n_dimesions} {alpha_indx}
#   EX: python fastElliptical.py 2**20 10 3 4

@jit(nopython=True)
def localEnergy(r,n_particles,n_dimensions,alpha,beta):
    """
    Expectation value for the energy of the system.
    """
    P = 0
    K = 0
    for k in range(n_particles):
        K += kineticEnergyParticle(r,n_particles,n_dimensions,alpha,r[k],k,hardCoreDiameter)
        square = 0
        for d in range(n_dimensions):
            if d == 2:
                square += (beta*r[k,d])**2
            else:
                square += r[k,d]**2
        P += 0.5*square
    return K + P

@jit(nopython=True)
def dE_L(r,n_particles,n_dimensions,alpha,beta):
    derivative = 0
    for i in range(n_particles):
        r_i = r[i]
        square = 0
        pos = 0
        for d in range(n_dimensions):
            if d == 2:
                pos += beta*r_i[d]
                square += (beta*r_i[d])**2
            else:
                pos += r_i[d]
                square += r_i[d]**2
        derivative += -4 -2*(beta + pos) + 8*alpha*square
    return derivative

@jit(nopython=True)
def innerProduct(r_k,r_l):
    prod = 0
    for d in range(r_k.shape[0]):
        prod += r_k[d]*r_l[d]
    return prod

@jit(nopython=True)
def kineticEnergyParticle(r,n_particles,n_dimensions,alpha,r_k,k,hardCoreDiameter):
    """
    Kinetic energy of the system
    """
    gradientSP  = gradientSingleParticle(r_k,n_dimensions,alpha,beta)
    gradientC   = gradientCorrelation(r,n_particles,n_dimensions,beta,r_k,k,hardCoreDiameter)
    laplaceSP   = laplaceSingleParticle(n_particles,n_dimensions,alpha,beta,r_k)
    laplaceC    = laplaceCorrelation(r,n_particles,n_dimensions,beta,r_k,k,hardCoreDiameter) + innerProduct(gradientC,gradientC)
    laplacian   = laplaceSP + laplaceC + 2*innerProduct(gradientSP,gradientC)
    return -0.5*(laplacian)

### WAVE FUNCTION
@jit(nopython=True)
def trialWave(r,n_particles,n_dimensions,alpha,beta,hardCoreDiameter):
    single = 0
    correlation = 1
    for i in range(n_particles):
        square = 0
        for d in range(n_dimensions):
            if d == 2:
                square += (beta*r[i,d])**2
            else:
                square += r[i,d]**2
        single -= alpha*square
        for j in range(i+1,n_particles):
            dist = 0
            for d in range(n_dimensions):
                if d == 2:
                    dist += (beta*r[i,d] - beta*r[j,d])**2
                else:
                    dist += (r[i,d] - r[j,d])**2
            r_ij = np.sqrt(dist)
            if r_ij > hardCoreDiameter:
                correlation *= 1 - hardCoreDiameter/(sqrt(r_ij)) # Correlation
            else:
                correlation *= 0
    return exp(single)*correlation  

### QUANTUM FORCE
@jit(nopython=True)
def quantumForce(r,n_particles,n_dimensions,alpha,beta,hardCoreDiameter):
    """
    Quantum force of the system
    Returns matrix.
    """
    qForce = np.zeros((n_particles,n_dimensions))
    vector = np.ones(n_dimensions)
    for k in range(n_particles):
        qForce[k] = 2 * (gradientSingleParticle(r[k],n_dimensions,alpha,beta) + gradientCorrelation(r,n_particles,n_dimensions,beta,r[k],k,hardCoreDiameter))
    return qForce

### Functions
@jit(nopython=True)
def monteRun(n_particles,n_dimensions,alpha,beta,hardCoreDiameter,
            posOld,posNew,trialNew,trialOld,qOld,qNew,
            dE,gradE,
            accepted,importanceSampling=True,D = 0.5):
    dE = 0
    for i in range(n_particles):
        if importanceSampling:
            greenOld = .0
            greenNew = .0
            greens = 0
        for j in range(n_dimensions):
            posNew[i,j] = posOld[i,j] + sqrt(stepSize)*np.random.normal(0.0,1.0) + D*stepSize*qOld[i,j]
        trialNew = trialWave(posNew,n_particles,n_dimensions,alpha,beta,hardCoreDiameter)
        qNew     = quantumForce(posNew,n_particles,n_dimensions,alpha,beta,hardCoreDiameter)
        
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
                dE       = localEnergy(posOld,n_particles,n_dimensions,alpha,beta)
                gradE = dE_L(posOld,n_particles,n_dimensions,alpha,beta)
        
        # Brute Force
        else:
            transRatio = trialNew**2 / trialOld**2
            if np.random.uniform(0.0,1.0) < transRatio:
                accepted += 1
                for j in range(n_dimensions):
                    posOld[i,j] = posNew[i,j]
                    qOld[i,j] = qNew[i,j]
                trialOld = trialNew
                dE       = localEnergy(posOld,n_particles,n_dimensions,alpha,beta)
                gradE = dE_L(posOld,n_particles,n_dimensions,alpha,beta)

    return dE,gradE,accepted

def MonteCarlo(n_runs,n_particles,n_dimensions,alpha,beta,stepSize):
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
    gradE = 0
    Egradient = 0
    MC_energies = np.zeros(n_runs)

    # INIT POSOLD
    for i in range(n_particles):
        for j in range(n_dimensions):
            posOld[i,j] = np.sqrt(stepSize) * (random() - .5)
        for j in range(i):
            dist = 0
            for d in range(n_dimensions):
                if d == 2:
                    dist += (beta*posOld[i,d]-beta*posOld[j,d])**2
                else:
                    dist += (posOld[i,d]-posOld[j,d])**2
            if np.sqrt(dist) < hardCoreDiameter:
               sys.exit('Error creating initialized positions')
    posNew = posOld
    trialOld = trialWave(posOld,n_particles,n_dimensions,alpha,beta,hardCoreDiameter)
    qOld = quantumForce(posOld,n_particles,n_dimensions,alpha,beta,hardCoreDiameter)
    # MONTECARLO
    accepted = 0

    for n in tqdm(range(n_runs)):
        dE,gradE,accepted = monteRun(n_particles,n_dimensions,alpha,beta,hardCoreDiameter,
            posOld,posNew,trialNew,trialOld,qOld,qNew,dE,gradE,accepted,importanceSampling=True)
        E  += dE
        E2 += dE**2
        MC_energies[n] = dE
        Egradient += gradE*dE
    E  /= n_runs
    E2 /= n_runs
    var = E2 - E**2

    print('Accepted:',accepted)
    return E,var,Egradient,MC_energies

if __name__ == '__main__':
    # System variables
    n_particles = int(sys.argv[2])
    n_dimensions = int(sys.argv[3])
    hardCoreDiameter = .00433
    # Simulation variables
    n_runs   = eval(sys.argv[1])
    alphs = np.array([0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.8])
    alpha   = alphs[int(sys.argv[4])]
    beta    = sqrt(8)
    stepSize = 0.005
    E,var,gradient,MC_energies = MonteCarlo(n_runs,n_particles,n_dimensions,alpha,beta,stepSize)

    ENERGY = E                    # Energy
    VARIANCE = var,               # Variance
    ERROR = np.sqrt(var/n_runs)   # Error
    df = DataFrame(ENERGY,index = np.array([alpha]),columns=['Energy'])
    df['Variance'] = VARIANCE
    df['Error'] = ERROR
    print(df)
    np.savetxt('data/EllipticalEnergies.txt',MC_energies)