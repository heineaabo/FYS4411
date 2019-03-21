from MonteCarlo import MonteCarlo
from SphericalSystem import System
import numpy as np
from   pandas import DataFrame 
from   time   import perf_counter as clock

runs = 2**10
timestep = .005 
particles = 2
dimensions = 3
variations = [-1,0,1]
alphas     = np.array([(.5 + .1*i) for i in variations])
betas = np.array([1])
system     = System(particles,dimensions)
print('System with {} particles in {} dimensions created!'.format(particles,dimensions))
simulation = MonteCarlo(system,runs,alphas,betas,stepSize=timestep)
print('Starting Monte Carlo simulation with {} runs!'.format(runs))
t1 = clock()
simulation.RunMultiple()
t2 = clock()
print('Time duration: {}s'.format(t2-t1))
simulation.PrintResults(save=False)
