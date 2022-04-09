import numpy as np
import scipy.special as sci
from multiprocessing import Pool, cpu_count
import emcee
import h5py
import write_gadget as wg

### Model parameters
G = 4.299581e+04       # kpc / 10^10 solar mass * (km/s)^2
r0 = 10.               # Scale Radius
q = 11.4               # Dispersion parameter
v0 = 230               # Circular velocity

### Sampling parameters
ndim = 3               # r, vr, vt
nwalkers = 32          # Number of MCMC walkers FIXME: for parallel maybe reduce?
nsamples = 4000        # Number of samples to be drawn for each walker
scatter_coeff = 0.1    # Amplitude of random scatter for walker initial pos
burnin = 100           # Number of burn-in steps
N = nwalkers*nsamples  # Number of active particles in simulation
print('Generating Mestel Disk with ' + "{:d}".format(N) + ' particles...')

### Cut functions (Zang 1976, Toomre 1989, see De Rijcke et al. 2019)
r_inner = 8.0
r_outer = 90.0
n       = 4
m       = 5
R_cut = 300.0          # Hard cut (set to much larger than active disk)

### Parallelism
use_parallel = False   # Use parallel sampling
nb_threads = 6         # Number of parallel threads to use

### IC File
fname = 'mestel.hdf5'  # Name of the ic file (dont forget .hdf5)
pickle_ics = 0         # Optional: Pickle ics (pos,vel,mass) for future use
box_size = 2000.0      # Size of simulation box
periodic_bdry = False  # Use periodic boundary conditions or not

# Compute other quantities from parameters
Sigma0 = v0**2 / (2.0*np.pi*G*r0) # Scale Surface density 
sig = v0 / np.sqrt(1.0+q) # Velocity dispersion
sig2 = v0**2 / (1.0+q)
F0 = Sigma0 / ( 2.0**(q/2.) * np.sqrt(np.pi) * r0**q * sig**(q+2.0) + sci.gamma((1+q)/2.0)) # DF Normalization factor
# Precompute quantities for efficiency during sampling
lnF0 = np.log(F0)
routv0m = (r_outer * v0)**m
lnroutv0m = np.log(routv0m)
rinv0n = (r_inner * v0)**n

### Helper Functions
# Binding potential / Binding energy
def relative_potential(r):
    return -v0**2 * np.log(r/r0)
def relative_energy(r,vr,vt):
    return relative_potential(r) - 0.5*(vr**2 + vt**2)

### Log of Distribution Function (De Rijcke et al. 2019, Sellwood 2012)
def log_prob(x):
    if (x[0] <= 0.0 or x[0] > R_cut ): return -np.inf
    elif x[2] < 0.0: return -np.inf
    else:
        rvt = x[0]*x[2]
        rvtn = rvt**n
        # FIXME: divide DF by r to obtain correct density, why?
        return np.log(rvtn) - np.log(rvtn + rinv0n) + lnF0 + q*np.log(rvt) + relative_energy(x[0],x[1],x[2])/sig2 + lnroutv0m - np.log(routv0m + rvt**m) + np.log(x[0])
    

### Initialize Walkers
startpos = np.array([r0,0.0,v0])
p0 = startpos + scatter_coeff * np.random.randn(nwalkers, len(startpos))

### Sample DF
if use_parallel:
    with Pool(processes=nb_threads) as pool: 
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, pool=pool)
        state = sampler.run_mcmc(p0, burnin)
        sampler.reset()
        sampler.run_mcmc(p0, nsamples,progress=True)
    samples = sampler.get_chain(flat=True)
    
else:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    state = sampler.run_mcmc(p0, burnin)
    sampler.reset()
    sampler.run_mcmc(state, nsamples,progress=True)
    samples = sampler.get_chain(flat=True)
     
### Convert samples to ICs
# Positions
theta_rand = np.random.uniform(0.0,2*np.pi,N)
x = samples[:,0]*np.cos(theta_rand)
y = samples[:,0]*np.sin(theta_rand)
z = np.zeros(N)

X = np.array([x,y,z]).transpose()

# Velocities
v_x = np.cos(theta_rand) * samples[:,1] - np.sin(theta_rand) * samples[:,2]
v_y = np.sin(theta_rand) * samples[:,1] + np.cos(theta_rand) * samples[:,2]
v_z = np.zeros(N)

V = np.array([v_x,v_y,v_z]).transpose()

### Generate Masses
# Integrate Mestel Density to obtain M(r) <-> r(M)
def m_of_r(r):
    return v0**2*r/G
# Total Mass of model = M(R_cut)
M_cut = m_of_r(R_cut)
m = M_cut/N * np.ones(N)

### Write to hdf5
print('Writing IC file...')
with h5py.File(fname,'w') as f:
    wg.write_header(
        f,
        boxsize=box_size,
        flag_entropy = 0,
        np_total = [0,N,0,0,0,0],
        np_total_hw = [0,0,0,0,0,0]
    )
    wg.write_runtime_pars(
        f,
        periodic_boundary = periodic_bdry
    )
    wg.write_units(
        f,
        length = 3.086e21,
        mass = 1.988e43,
        time = 3.086e21 / 1.0e5,
        current = 1.0,
        temperature = 1.0
    )
    wg.write_block(
        f,
        part_type = 1,
        pos = X,
        vel = V,
        mass = m,
        ids=np.arange(N), # Overridden by params.yml
        int_energy = np.zeros(N),
        smoothing = np.ones(N)
    )
print('done.')
#print('Generated Truncated Mestel disk with cuts:')
#print('M_inner = ' + "{:.1e}".format(M_cut_inner) + '   -->   R_inner = ' + "{:.1e}".format(R_cut_inner))
#print('M_outer = ' + "{:.1e}".format(M_cut) + '   -->   R_outer = ' + "{:.1e}".format(R_cut))
epsilon0 = np.sqrt(2.0*G*M_cut*r0)/v0 / np.sqrt(N)
print('Recommended Softening length (times conventional factor): ' + "{:.4e}".format(epsilon0) + ' kpc')
    
"""
import matplotlib.pyplot as plt
bins = 400
r_mestel = np.random.uniform(0.0,R_cut,5*N)
diff = np.histogram(r_mestel,bins=bins)[0] - np.histogram(samples[:,0],bins=bins)[0]
#plt.hist(r_mestel - samples[:,0],400)
rspace = np.linspace(0.0,R_cut,bins)
plt.plot(rspace,diff)
plt.show()
"""

"""
from matplotlib import pyplot as plt
figsize = 8
dotsize = 0.05
lim = 90
fig, ax = plt.subplots(figsize=(figsize,figsize))
ax.scatter(x,y,s=dotsize,c='white')
ax.set_facecolor('black')
ax.set_aspect('equal', 'box')
ax.set_xlim(-lim,lim)
ax.set_ylim(-lim,lim)
ax.set_xlabel('x [kpc]')
ax.set_ylabel('y [kpc]')
plt.show()
"""

"""
from matplotlib import pyplot as plt
figsize = 8
dotsize = 0.05
lim = 90
fig, ax = plt.subplots(figsize=(figsize,figsize))
ax.quiver(x,y,v_x,v_y,color='white')
ax.set_facecolor('black')
ax.set_aspect('equal', 'box')
ax.set_xlim(-lim,lim)
ax.set_ylim(-lim,lim)
ax.set_xlabel('x [kpc]')
ax.set_ylabel('y [kpc]')
plt.show()
"""
