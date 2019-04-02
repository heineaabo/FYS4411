import numpy as np
from tqdm import tqdm
from numba import njit
from random import random
import sys

@njit
def MonteCarloSingle(n_particles,n_dimensions, alpha,beta,timeStep,hardCoreDiameter=.00433,D=0.5,MetropolisHastings=True):
    """
    Run single Monte Carlo cycle
    """
    
    posOld,posNew, qOld,qNew, trialOld,trialNew = Initialize(n_particles,n_dimensions,alpha,beta,timeStep,hardCoreDiameter)
    Energy = 0
    EnergySquared = 0
    sampleEnergy = 0
    accepted = 0
    runs = 0
    for i in range(n_particles):
        runs += 1 # for acceptance ratio
        for j in range(n_dimensions): 

            posNew[i,j] = posOld[i,j] + np.random.uniform(0.0,1.0)#np.sqrt(timeStep)*np.random.normal(0.0,1.0) + D*timeStep*qOld[i,j]
        trialNew = trialWave(posNew, n_particles,n_dimensions,alpha,beta,hardCoreDiameter)
        qNew     = quantumForce(posNew, n_particles,n_dimensions,alpha,beta,hardCoreDiameter)
    
        Greens = 1 # if brute force metropolis
        
        if MetropolisHastings==True: # if Metropolis-Hastings
            green_x_y = .0
            green_y_x = .0
            for j in range(n_dimensions):
                green_x_y -= (posNew[i,j] - posOld[i,j] -D*timeStep*qOld[i,j])**2
                green_y_x -= (posOld[i,j] - posNew[i,j] -D*timeStep*qNew[i,j])**2
            green_x_y /= 4*D*timeStep
            green_y_x /= 4*D*timeStep
            Greens = np.exp(green_x_y-green_y_x)
        
        transitionRatio = Greens*trialNew**2 / trialOld**2

        if np.random.uniform(0.0,1.0) < transitionRatio: # Accept move
            for j in range(n_dimensions):
                posOld[i,j] = posNew[i,j]
                qOld[i,j] = qNew[i,j]
            trialOld = trialNew
            accepted += 1
    return posOld,(accepted/runs)

#@njit
def MonteCarloMultiple(n_runs,n_particles,n_dimensions, alpha,beta,timeStep,hardCoreDiameter=.00433,D=0.5,MetropolisHastings=True):
    """
    Run all Monte Carlo cycles.
    """
    EnergyTotal = 0
    EnergySquaredTotal = 0

    EnergyList = np.zeros(n_runs)
    for n in tqdm(range(n_runs)):
        posOld,acceptratio = MonteCarloSingle(n_particles,n_dimensions, alpha,beta,timeStep,hardCoreDiameter,D,MetropolisHastings)
        sampleEnergy = localEnergy(posOld, n_particles,n_dimensions,alpha,beta,hardCoreDiameter)
        EnergyTotal += sampleEnergy
        EnergySquaredTotal += sampleEnergy**2
        EnergyList[n] = sampleEnergy
    
    EnergyTotal /= n_runs
    EnergySquaredTotal /= n_runs
    Variance = EnergySquaredTotal - EnergyTotal**2

    return EnergyTotal,Variance,EnergyList
@njit
def Initialize(n_particles,n_dimensions,alpha,beta,timeStep,hardCoreDiameter):
    """
    Initialize the simulation matrices and variables.
    """
    posOld = initPosition(n_particles,n_dimensions,beta,timeStep,hardCoreDiameter) # Assign random postitions
    posNew = posOld.copy()
    qOld     = quantumForce(posOld, n_particles,n_dimensions,alpha,beta,hardCoreDiameter) # qOld = np.zeros((n_particles,n_dimensions))
    qNew     = np.zeros((n_particles,n_dimensions))
    trialOld = trialWave(posOld, n_particles,n_dimensions,alpha,beta,hardCoreDiameter) 
    trialNew = 0

    return posOld,posNew, qOld,qNew, trialOld,trialNew

@njit
def initPosition(n_particles,n_dimensions,beta,timeStep,hardCoreDiameter=.00433):
    posOld = np.zeros((n_particles,n_dimensions))
    for i in range(n_particles):
        for j in range(n_dimensions):
            posOld[i,j] = np.sqrt(timeStep) * np.random.normal(0.0,1.0)
        for k in range(i):
            dist = 0
            for d in range(n_dimensions):
                    dist += (posOld[i,d]-posOld[k,d])**2
            if np.sqrt(dist) < hardCoreDiameter:
                return initPosition(n_particles,n_dimensions,beta,hardCoreDiameter)
    return posOld

def GradientDescent(n_runs,n_particles,n_dimensions, alpha,timeStep,D,eta,n_iter,limit, MetropolisHastings,trap='S'):
    energy,var,alphaDer = MonteCarloMultiple(n_runs,n_particles,n_dimensions, alpha,timeStep,D, MetropolisHastings=True,trap='S')
    alphaOld = alpha
    for i in range(n_iter):
        print('Iteration {}: Energy = {:.5f}, grad = {:.5f}, alpha = {}'.format(i,energy,alphaDer,alphaOld))
        gradE = alphaDer
        alphaNew = alphaOld - eta*gradE
        if np.abs(alphaNew - alphaOld) < limit:
            print('Limit reached for:',alphaNew - alphaOld )
            break
        energy,var,alphaDer = MonteCarloMultiple(n_runs,n_particles,n_dimensions, alphaNew,timeStep,D, MetropolisHastings=True,trap='S')
        alphaOld = alphaNew    
    print('Minimized for alpha={}'.format(alphaNew))

if __name__ == '__main__':
    trap = sys.argv[1]
    if trap == 'S':
        print('trap potential (S)')
        from Hamiltonian import HamiltonianSpherical  as localEnergy
        from WaveFunction import trialWaveSpherical as trialWave
        from QuantumForce import quantumForceSpherical as quantumForce
        from WaveFunction import dPsiSpherical as dPsi
        beta = 1
        hardCoreDiameter= .0
    if trap == 'E':
        print('trap potential (E)')
        from Hamiltonian import HamiltonianElliptical as localEnergy
        from WaveFunction import trialWaveElliptical as trialWave
        from QuantumForce import quantumForceElliptical as quantumForce
        from WaveFunction import dPsiElliptical  as dPsi
        beta = np.sqrt(8)
        hardCoreDiameter=.00433
    
    n_runs = sys.argv[2]
    n_particles = sys.argv[3]
    n_dimensions = sys.argv[4]
    alphas = sys.argv[5]
    timeStep = 0.01
    D=0.5
    Energy ,Variance, EnergyList= MonteCarloMultiple(n_runs,n_particles,n_dimensions, alpha,beta,timeStep,hardCoreDiameter,D, MetropolisHastings=False)
    print(Energy,Variance)

