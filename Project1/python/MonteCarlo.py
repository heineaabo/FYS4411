import numpy  as     np
from   random import random,seed,normalvariate
from   math   import exp,sqrt
from   pandas import DataFrame 
from   time   import perf_counter as clock
from tqdm import tqdm
from sys import argv
from tools import *

# Class imports
#from   System import System
#from   resampling import *

class MonteCarlo():
    def __init__(self, 
                System,     # System class
                N,          # Number of Monte Carlo runs
                alphaList,  # numpy.array with values of alpha
                betaList,   # numpy.array with values of beta
                stepSize):  # Step length

        # System
        self.System       = System
        self.trialWave    = System.trialWave
        self.localEnergy  = System.localEnergy
        self.quantumForce = System.quantumForce
        self.n_particles  = System.n_particles
        self.n_dimensions = System.n_dimensions
        self.hardCoreDiameter = System.hardCoreDiameter

        # Simulation variables
        self.n_runs   = N
        self.alphas   = alphaList
        self.betas    = betaList
        self.stepSize = stepSize
        # Simulation matrices
        self.posOld   = np.zeros((self.n_particles,self.n_dimensions))
        self.posNew   = np.zeros((self.n_particles,self.n_dimensions))
        self.trialNew = 0 # np.zeros((self.n_particles,self.n_dimensions))
        self.trialOld = 0 # np.zeros((self.n_particles,self.n_dimensions))
        self.qNew     = np.zeros((self.n_particles,self.n_dimensions))
        self.qOld     = np.zeros((self.n_particles,self.n_dimensions))

        # Energies and statistics
        self.E  = 0
        self.E2 = 0
        self.dE = 0
        self.MC_energies = np.zeros(N)
        self.energies = np.zeros((self.alphas.shape[0],self.betas.shape[0]))
        self.variance = np.zeros((self.alphas.shape[0],self.betas.shape[0]))
        self.error    = np.zeros((self.alphas.shape[0],self.betas.shape[0]))

    def initializePosition(self):
        position = np.zeros((self.n_particles,self.n_dimensions))
        for i in range(self.n_particles):
            for j in range(self.n_dimensions):
                position[i,j] = np.sqrt(self.stepSize) * (random() - .5)
            #position[i] = np.sqrt(self.stepSize) * np.random.uniform(0.0,1.0,self.n_dimensions)
            for j in range(i):
                if relativeDistance(position[i],position[j]) < self.hardCoreDiameter:
                    print('Wrong')
        return position
    
    def Initialize(self):
        self.E  = 0
        self.E2 = 0
        self.dE = 0
        self.posOld = self.initializePosition()
        self.posNew = self.posOld
        self.trialOld = self.trialWave(self.posOld)
        self.qOld     = self.quantumForce(self.posOld)
        #print(self.qOld)

    def RunSingle(self,D=0.5,importanceSampling=True):
        #runs = self.n_runs
        accepted = 0
        positive = 0
        negative = 0
        for n in tqdm(range(self.n_runs)):
            for i in range(self.n_particles):
                if importanceSampling:
                    greenOld = .0
                    greenNew = .0
                for j in range(self.n_dimensions):
                    self.posNew[i,j] = self.posOld[i,j] + sqrt(self.stepSize)*normalvariate(0.0,1.0) + D*self.stepSize*self.qOld[i,j]

                self.trialNew = self.trialWave(self.posNew)
                self.qNew     = self.quantumForce(self.posNew)
                
                # Metropolis-Hasting
                if importanceSampling:
                    for j in range(self.n_dimensions):
                        greenOld -= (self.posNew[i,j] - self.posOld[i,j] -D*self.stepSize*self.qOld[i,j])**2
                        greenNew -= (self.posOld[i,j] - self.posNew[i,j] -D*self.stepSize*self.qNew[i,j])**2
                    Greens = exp(greenOld)/exp(greenNew)

                    transRatio = Greens * self.trialNew**2 / self.trialOld**2
                        
                    if np.random.uniform(0.0,1.0) < transRatio: #random()
                        accepted += 1
                        for j in range(self.n_dimensions):
                            self.posOld[i,j] = self.posNew[i,j]
                            self.qOld[i,j] = self.qNew[i,j]
                        #self.posOld   = self.posNew.copy()
                        self.trialOld = self.trialNew
                        #self.qOld     = self.qNew.copy()
                        self.dE       = self.localEnergy(self.posOld)
                        if self.dE >= 0:
                          positive += 1
                        if self.dE < 0:
                           negative += 1
                
                # Brute Force
                else:
                    transRatio = self.trialNew**2 / self.trialOld**2
                    if np.random.uniform(0.0,1.0) < transRatio:
                        accepted += 1
                        # for j in range(self.n_dimensions):
                        #     self.posOld[i,j] = self.posNew[i,j]
                        #     self.qOld[i,j] = self.qNew[i,j]
                        self.posOld   = self.posNew.copy()
                        self.trialOld = self.trialNew
                        self.qOld     = self.qNew.copy()
                        self.dE       = self.localEnergy(self.posOld)
                        if self.dE >= 0:
                          positive += 1
                        if self.dE < 0:
                           negative += 1

            self.E  += self.dE
            self.E2 += self.dE**2
            self.MC_energies[n] = self.E
            #print(self.dE,self.dE**2)
        self.E  /= self.n_runs
        self.E2 /= self.n_runs
        var = self.E2 - self.E**2
        print('Accepted:',accepted)
        print('Positive:{}, Negative: {}'.format(positive,negative))

        return(self.E,                    # Energy
                var,                      # Variance
                np.sqrt(var/self.n_runs)) # Error

    def RunMultiple(self,importanceSampling=True):
        for a,alpha in enumerate(self.alphas):
            self.System.alpha = alpha
            print('Alpha = {}, Progress: {}/{}'.format(alpha,a+1,self.alphas.shape[0]))
            for b,beta in enumerate(self.betas):
                self.System.beta = beta
                self.Initialize()
                self.energies[a,b],self.variance[a,b],self.error[a,b] = self.RunSingle(importanceSampling=importanceSampling)

    def FindOptimalParameters(self,iterMax=100,learnRate=.01): # Using Conjugate gradient descent
        self.Initialize()
        for i in range(iterMax):
            energy = self.RunSingle(runs=1000)
            gradient = something
            self.alpha = -learnRate*gradient
        return energy

    def PrintResults(self, save=False):
        df = DataFrame(self.energies,index = self.alphas,columns=['Energy'])
        df['Variance'] = self.variance
        df['Error'] = self.error
        print(df)
        if save:
            np.savetxt('data/energy.txt',self.MC_energies)
            #from datetime import datetime
            #dato = str(datetime.now())
            #date = '{}-{}-{}'.format(dato[8:10],dato[5:7],dato[0:4])
            #time = '{}:{}:{}'.format(dato[11:13],dato[14:16],dato[17:19])
            #df.to_pickle('data/{}_{}.pkl'.format(date,time))

    
if __name__ == '__main__':
    trap = argv[1]
    variation = int(argv[2])
    if trap not in ['S','E']:
        raise ValueError('Specify trap potential: "S" or "E"')
    runs = int(argv[3])
    timestep = eval(argv[4])
    variations = np.arange(variation) - np.mean(np.arange(variation))
    particles = 10
    dimensions = 3
    alphas     = np.array([(.5 + .05*i) for i in variations])
    #alphas = np.array([0.5])
    if trap == 'S':
        betas = np.array([1])
        from SphericalSystem import System
    if trap == 'E':
        betas = np.array([np.sqrt(8)])
        from EllipticalSystem import System
    system     = System(particles,dimensions)
    print('System with {} particles in {} dimensions created!'.format(particles,dimensions))
    simulation = MonteCarlo(system,runs,alphas,betas,stepSize=timestep)
    print('Starting Monte Carlo simulation with {} runs!'.format(runs))
    t1 = clock()
    simulation.RunMultiple(importanceSampling=True)
    t2 = clock()
    print('Time duration: {}s'.format(t2-t1))
    simulation.PrintResults()
