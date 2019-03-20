import numpy  as     np
from   math   import exp,sqrt
from   random import random
from   tools import *

#from   Hamiltonian  import Hamiltonian
#from   WaveFunction import WaveFunction

class System():
    def __init__(self,n_particles,n_dimensions):
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.alpha = 0
        self.beta = 0
        self.hardCoreDiameter = .00433
        self.trialWave = self.trialWave
        self.localEnergy = self.Hamiltonian
        self.quantumForce = self.quantumForce


    ### HAMILTONIAN
    def Hamiltonian(self,r):
        """
        Expectation value for the energy of the system.
        """
        P = 0
        K = 0
        for k in range(self.n_particles):
            K += self.kineticEnergyParticle(r,r[k],k)
            P += 0.5*self.squarePosition(r[k])
        return K + P

    def kineticEnergyParticle(self,r,r_k,k):
        """
        Kinetic energy of the system
        """
        R = self.distancesMatrix(r)
        gradientSP  = self.gradientSingleParticle(r_k)
        gradientC   = self.gradientCorrelation(r,r_k,k)
        laplaceSP   = self.laplaceSingleParticle(r_k)
        laplaceC    = self.laplaceCorrelation(r,r_k,k) + np.inner(gradientC,gradientC)
        laplacian   = laplaceSP + laplaceC + 2*np.inner(gradientSP,gradientC)
        return -0.5*(laplacian)

    ### WAVE FUNCTION
    def trialWave(self,r):
        single = 0
        correlation = 1
        R = self.distancesMatrix(r)
        for i in range(self.n_particles):
            single -= self.alpha*self.squarePosition(r[i]) # Singel particle
            for j in range(i+1,self.n_particles):
                r_ij = R[i,j]
                if r_ij > self.hardCoreDiameter:
                    correlation *= 1 - self.hardCoreDiameter/(sqrt(r_ij)) # Correlation
                else:
                    correlation *= 0
        return exp(single)*correlation

    ### QUANTUM FORCE
    def quantumForce(self,r):
        """
        Quantum force of the system
        Returns matrix.
        """
        R = self.distancesMatrix(r)
        qForce = np.zeros((self.n_particles,self.n_dimensions))
        vector = np.ones(self.n_dimensions)
        for k in range(self.n_particles):
            qForce[k] = 2 * (self.gradientSingleParticle(r[k]) + self.gradientCorrelation(r,r[k],k))
        return qForce

    ### Functions  
    def gradientSingleParticle(self,r_k):
        """
        Gradient of single-particle function.
        Returns vector with elements for each dimension.
        """
        gradient = np.zeros(self.n_dimensions)
        for d in range(self.n_dimensions):
            if d == 2:
                gradient[d] += self.beta*r_k[d]
            else:
                gradient[d] += r_k[d]
        return -2*self.alpha*gradient
    
    def laplaceSingleParticle(self,r_k):
        """
        Laplacian of the single-particle function.
        Returns scalar.
        """
        gradient = self.squarePosition(r_k)
        return -2*self.alpha*(2 + self.beta) + 4*(self.alpha**2)*gradient 

    def gradientCorrelation(self,r,r_k,k):
        """
        Gradient of correlation function.
        Returns vector
        """
        gradient = np.zeros(self.n_dimensions)
        for l in range(self.n_particles):
            if l == k:
                continue
            nominator = self.vectorDistance(r_k,r[l])
            r_kl = self.euclideanDistance(r_k,r[l])
            derivative = self.du(r_kl)
            for d in range(self.n_dimensions):
                gradient[d] = (nominator[d]/r_kl)*derivative
        return gradient
    
    def laplaceCorrelation(self,r,r_k,k):
        """
        Laplacian of correlation function.
        Returns scalar.
        """
        laplace = 0
        for l in range(self.n_particles):
            if l == k:
                continue
            r_kl = self.euclideanDistance(r_k,r[l])
            laplace += (2/r_kl)*self.du(r_kl)+ self.du2(r_kl)
        return laplace

    def du(self,r_kl):
        """
        Derivative of the correlation function for particles r_k and r_l.
        """
        a = self.hardCoreDiameter
        return a/(r_kl*(r_kl-a))
    def du2(self,r_kl):
        """
        Second derivative of the correlation function for particles r_k and r_l.
        """
        a = self.hardCoreDiameter
        return (a**2 - 2*a*r_kl)/((r_kl**2)*(r_kl-a)**2)
    
    def squarePosition(self,r_k):
        """
        Squared position, takes into account the beta-parameter of the z-value.
        """
        square = 0
        for d in range(self.n_dimensions):
            if d == 2:
                square += (self.beta*r_k[d])**2
            else:
                square += r_k[d]**2
        return square
    def squarePositionVector(self,r_k):
        """
        Squared position, takes into account the beta-parameter of the z-value.
        """
        square = np.zeros(self.n_dimensions)
        for d in range(self.n_dimensions):
            if d == 2:
                square[d] = (self.beta*r_k[d])**2
            else:
                square[d] = r_k[d]**2
        return square

    def vectorDistance(self,r_k,r_l):
        dist = np.zeros(self.n_dimensions)
        for d in range(self.n_dimensions):
            if d == 2:
                dist[d] = (self.beta*r_k[d]-self.beta*r_l[d])
            else:
                dist[d] = (r_k[d]-r_l[d])
        return dist
    def euclideanDistance(self,r_k,r_l):
        dist = 0
        for d in range(self.n_dimensions):
            if d == 2:
                dist += (self.beta*r_k[d]-self.beta*r_l[d])**2
            else:
                dist += (r_k[d]-r_l[d])**2
        return sqrt(dist)

    def distancesMatrix(self,r):
        distances = np.zeros((self.n_particles,self.n_particles))
        for i in range(self.n_particles):
            for j in range(i+1,self.n_particles):
                distances[i,j] = relativeDistance(r[i],r[j])
        return distances
