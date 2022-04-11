import numpy as np
import scipy.special as sci
from multiprocessing import Pool, cpu_count
import emcee
import h5py
import write_gadget as wg
from scipy.integrate import quad, dblquad
from scipy.interpolate import interp1d
from tqdm import tqdm

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

### Cut functions (Zang 1976, Toomre 1989, see De Rijcke et al. 2019)
r_inner = 5.0
r_outer = 11.5*r_inner
n       = 4
m       = 5
R_cut = 20.0*r_inner          # Hard cut (set to much larger than active disk)
#chi     = 1          # Fraction of total mass to be active

## Missing Mass sampling
nwalkers_mm = 32
nsamples_mm = 1000
N_mm = nwalkers_mm * nsamples_mm # Number of unresponsive background particles to compensate missing mass
active_id_start = 1000000 # Particle ID below which the particles are unresponsive (see https://swift.dur.ac.uk/docs/GettingStarted/special_modes.html)

### Parallelism
use_parallel = False   # Use parallel sampling
nb_threads = 6         # Number of parallel threads to use

### IC File
fname = 'mestel_new.hdf5'  # Name of the ic file (dont forget .hdf5)
pickle_ics = 0         # Optional: Pickle ics (pos,vel,mass) for future use
box_size = 2000.0      # Size of simulation box
periodic_bdry = False  # Use periodic boundary conditions or not

# Compute other quantities from parameters
Sigma0 = v0**2 / (2.0*np.pi*G*r0) # Scale Surface density 
sig = v0 / np.sqrt(1.0+q) # Velocity dispersion
sig2 = v0**2 / (1.0+q)
F0 = Sigma0 / ( 2.0**(q/2.) * np.sqrt(np.pi) * r0**q * sig**(q+2.0) * sci.gamma((1+q)/2.0)) # DF Normalization factor
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
        # Truncated Disk
        return np.log(rvtn) - np.log(rvtn + rinv0n) + lnF0 + q*np.log(rvt) + relative_energy(x[0],x[1],x[2])/sig2 + lnroutv0m - np.log(routv0m + rvt**m) + np.log(x[0])
        # Untruncated Disk
        #return lnF0 + q*np.log(rvt) + relative_energy(x[0],x[1],x[2])/sig2 + np.log(x[0])
    

### Initialize Walkers
startpos = np.array([r0,0.0,v0])
p0 = startpos + scatter_coeff * np.random.randn(nwalkers, len(startpos))

### Sample DF
print('Generating Truncated Mestel Disk with ' + "{:d}".format(N) + ' particles...')
if use_parallel:
    with Pool(processes=nb_threads) as pool: 
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, pool=pool)
        state = sampler.run_mcmc(p0, burnin)
        sampler.reset()
        sampler.run_mcmc(state, nsamples,progress=True)
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

'''
####### Compute Missing mass
The distribution function that was sampled so far contains the 2 truncation functions, and hence
does not generate a surface density equal to the Mestel density. To compensate for this, the
"missing mass density" is calculated numerically and sampled to produce another set of particles
that will be static (or "unresponsive") in the simulation and act merely as a kind of background
potential.

The mass density of the truncated model (which is not analytical) also must be known to calculate
the total mass and hence the masses of the (active) particles, this is done below.
'''
print('Computing mass density of truncated disk, missing mass density and integrate to get total mass...')
# Truncated Distribution Function (same as above, without log)
def DF(vr,vt,r):
    L = r*vt
    return F0 * L**(n+q) * routv0m / ((rinv0n+L**n)*(routv0m+L**m)) * np.exp(relative_energy(r,vr,vt)/sig2)

# Truncated DF marginalized (integrated) over vr (analytically), for efficiency
def DF_intover_vr(vt,r):
    L = r*vt
    return F0 * routv0m * r**(n-1) * vt**(n+q) * np.exp(-0.5*vt**2/sig2) / ((rinv0n + L**n)*(routv0m + L**m)) * np.sqrt(2*np.pi)*sig / r0**(-1-q)

# Numerically integrate over vr to produce Sigma(r)
def density_truncated(r):
    return quad(DF_intover_vr,0.0,np.inf,args=(r,))[0]

# Numerically integrate over vr and r to produce M(r)
def integrand(vt,r):
    return 2*np.pi*r*DF_intover_vr(vt,r)
def m_of_r_truncated(r):
    return dblquad(integrand,0.0,r,0.0,np.inf)[0]

# Mestel density
def density_mestel(r):
    return v0**2/(2.0*np.pi*G*r)

# Total Mass at R_cut --> mass of particles
# Theoretical total mass of Mestel model
Mtot = v0**2 * R_cut / G

# Active Disk
M = m_of_r_truncated(R_cut)
m = M / N

# Missing Mass
M_mm = Mtot - M
m_mm = M_mm / N_mm

print('Total Mass of untruncated disk: ' + "{:.2e}".format(Mtot*1e10) + ' solar masses.')
print('Total Mass of truncated disk:   ' + "{:.2e}".format(M*1e10) + ' solar masses.')
print('Missing Mass:                   ' + "{:.2e}".format(M_mm*1e10) + ' solar masses.')

# Create more efficient truncated density via interpolation
nb_radii_interp = 500
interp_rspace = np.linspace(0.0,R_cut,nb_radii_interp)
interp_dens = np.empty(nb_radii_interp)
for i,r in enumerate(interp_rspace):
    interp_dens[i] = density_truncated(r)
density_truncated_interp = interp1d(interp_rspace,interp_dens)

# Missing mass density
def missing_mass_density(r):
    return density_mestel(r) - density_truncated_interp(r)

# Sample missing mass
def logprob_mm(r):
    if r <= 0.0 or r > R_cut : return -np.inf
    else: return np.log(missing_mass_density(r)) + np.log(r)
print('Generating ' + "{:d}".format(N_mm) + ' background particles to compensate for missing mass...')
sampler_mm = emcee.EnsembleSampler(nwalkers_mm, 1, logprob_mm)
startpos_mm = np.array(r0)
p0_mm = startpos_mm + scatter_coeff * np.random.randn(nwalkers_mm, 1)
state_mm = sampler_mm.run_mcmc(p0_mm, burnin)
sampler_mm.reset()
sampler_mm.run_mcmc(state_mm, nsamples_mm,progress=True)
samples_mm = sampler_mm.get_chain(flat=True)

# Convert to cartesian
r_mm = samples_mm[:,0]
theta_rand_mm = np.random.uniform(0.0,2*np.pi,N_mm)
x_mm = r_mm*np.cos(theta_rand_mm)
y_mm = r_mm*np.sin(theta_rand_mm)
z_mm = np.zeros(N_mm)

X_mm = np.array([x_mm,y_mm,z_mm]).transpose()

# Merge Active & Passive particles to write IC file
X_full = np.concatenate((X,X_mm))
V_full = np.concatenate((V,np.zeros((N_mm,3))))
M_full = np.concatenate((m*np.ones(N),m_mm*np.ones(N_mm)))
IDs    = np.concatenate((np.arange(active_id_start,N),np.arange(N_mm)))

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
    # Active Particles
    #wg.write_block(
    #    f,
    #    part_type = 1,
    #    pos = X,
    #    vel = V,
    #    mass = m*np.ones(N),
    #    ids=np.arange(active_id_start,N),
    #    int_energy = np.zeros(N),
    #    smoothing = np.ones(N)
    #)

    wg.write_block(
        f,
        part_type = 1,
        pos = X_full,
        vel = V_full,
        mass = M_full,
        ids = IDs,
        int_energy = np.zeros(N+N_mm),
        smoothing = np.ones(N+N_mm)
    )
print('done.')