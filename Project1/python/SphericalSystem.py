import numpy  as     np
from   math   import exp,sqrt
from   random import random
from tools import *

#from   Hamiltonian  import Hamiltonian
#from   WaveFunction import WaveFunction

class System():
    def __init__(self,n_particles,n_dimensions):
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.alpha = 0
        self.beta = 1
        self.hardCoreDiameter = 0
        self.trialWave = self.trialWave
        self.localEnergy = self.Hamiltonian
        self.quantumForce = self.quantumForce


    ### HAMILTONIAN
    def Hamiltonian(self,r):
        oneBody = self.alpha*self.n_dimensions*self.n_particles
        C = (0.5 - 2*self.alpha**2)
        for i in range(self.n_particles):
            oneBody += C*self.squarePosition(r[i])
        return oneBody

    ### WAVE FUNCTION
    def trialWave(self,r):
        wave = 0
        for i in range(self.n_particles):
            wave -= self.alpha*self.squarePosition(r[i])
            #for j in range(self.n_dimensions):
            #    wave -= self.alpha*(r[i,j]**2) # Single particle
        return exp(wave)

    ### QUANTUM FORCE
    def quantumForce(self,r):
        return -4*self.alpha*r

    ### Gradient
    def gradient(self,r_k,vector=False):
        """
        Gradient of single-particle function.
        Returns vector with elements for each dimension.
        """
        gradient = np.zeros(self.n_dimensions)
        for d in range(self.n_dimensions):
            if d == 2:
                gradient[d] += r_k[d]
            else:
                gradient[d] += self.beta*r_k[d]
        if vector:
            return -2*self.alpha*gradient
        else:
            return -2*self.alpha*np.sum(gradient)

    ### Functions            
    def squarePosition(self,r_k,vector=False):
        """
        Squared position, takes into account the beta-parameter of the z-value.
        """
        square = np.zeros(self.n_dimensions)
        for d in range(self.n_dimensions):
            square[d] = r_k[d]**2
        if vector:
            return square
        else:
            return np.sum(square)