################################################################################
# Copyright (c) 2022 Patrick Hirling (patrick.hirling@epfl.ch)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
################################################################################

import numpy as np
import scipy.special as sci
from scipy.optimize import minimize
from tqdm import tqdm
import time
import h5py
import write_gadget as wg

####### Parameters
# Plummer Model
G = 4.299581e+04  # Gravitational constant [kpc / 10^10 M_s * (kms)^2]
a = 0.05          # Plummer softening length [kpc]
M = 1.0e-5        # Total Mass [10^10 M_s]
q = 0.0			  # Anisotropy Parameter (-inf,2]

# IC File
N = 1000000        # Number of Particles
bound = 2.0		  # Max distance to origin (exclude outliers)
fname = 'plummer.hdf5' # Name of the ic file (dont forget .hdf5)
pickle_ics = 0      # Optional: Pickle ics (pos,vel,mass) for future use
box_size = 4.0
periodic_bdry = 0

# Parallelism for velocity sampling
use_parallel = True
nb_threads = 6  # Set to None to use os.cpucount()

if (q==0.0): print('WARNING: For q=0, use isotropic plummer script for better performance')

####### Generate Positions

def r_of_m(m,a,M):
    return a * ((M/m)**(2./3.) - 1)**(-1./2.)

m_rand = M*np.random.uniform(0.0,1.0,N)
r_rand = r_of_m(m_rand,a,M)
phi_rand = np.random.uniform(0.0,2*np.pi,N)
theta_rand = np.arccos( np.random.uniform(-1.,1.,N) )

x = r_rand * np.sin(theta_rand) * np.cos(phi_rand)
y = r_rand * np.sin(theta_rand) * np.sin(phi_rand)
z = r_rand * np.cos(theta_rand)

X = np.array([x,y,z]).transpose()

####### Dejonghe (1987) Anisotropic Distribution Function for Plummer Model

normalization = 3.0*sci.gamma(6.0-q) / (2.0*(2.0*np.pi)**(2.5))
units = G**(q-5.0) * M**(q-4.0) * a**(2.0-q)

def H(E,L):
    x = L**2 / (2.0*E*a**2)
    if x<=1.0:
        return 1.0/(sci.gamma(4.5-q)) * sci.hyp2f1(0.5*q,q-3.5,1.0,x)
    else:
        return 1.0/(sci.gamma(1.0-0.5*q)*sci.gamma(4.5-0.5*q)*(x**(0.5*q))) *sci.hyp2f1(0.5*q,0.5*q,4.5-0.5*q,1.0/x)

def Fq(E,L):
    if E < 0.0:
        return 0.0
    else:
        return normalization * units * E**(3.5-q) * H(E,L) 

####### Helper Functions
# Total energy: E = phi - 1/2v^2. relative potential: psi = -phi. Relative energy: -E = psi - 1/2v^2
def relative_potential(r): # = - phi
    return G*M / np.sqrt(r**2 + a**2)

def relative_energy(r,vr,vt):
    return relative_potential(r) - 0.5 * (vr**2 + vt**2)

# N.B: Angular momentum: L = r * vt

# Convenience Function for scipy.minimize negative of Fq*vt, indep. vars passed as array
def Fq_tomin(v,r):
    return -Fq(relative_energy(r,v[0],v[1]),r*v[1])*v[1]

####### Find max of DF at given radius
def fmax(r,vmax):
    #vmax = np.sqrt(2. * relative_potential(r))
    
    args = (r,)
    
    # Constraint function (defined locally, dep on r)
    def vel_constr2(v):
        return vmax**2-v[0]**2-v[1]**2
    
    # Initial Guess
    v0 = [0.1*vmax,0.2*vmax]

    # Constraint Object
    cons = ({'type':'ineq', 'fun': vel_constr2})
    
    # Minimize through scipy.optimize.minimize
    #fm = minimize(Fq_tomin,v0,constraints=cons,method = 'COBYLA',args=args)
    fm = minimize(Fq_tomin,v0,constraints=cons,method = 'SLSQP',args=args,bounds=[(0,vmax),(0,vmax)])
    
    # Min of negative df == max of df
    return -fm.fun

# Sample vr,vt from DF at given Radius
def sample_vel(r):
    # Compute max velocity (equiv. condition for E>=0)
    vmax = np.sqrt(2. * relative_potential(r))
    # Compute max of DF at this radius
    fm = 1.1*fmax(r,vmax) # 1.1 factor to be sure to include max
    while True:
        # Select candidates for vr,vt based on max full velocity
        while True:
            vr = np.random.uniform(0.0,vmax)
            vt = np.random.uniform(0.0,vmax)
            if (vr**2 + vt**2 <= vmax**2): break
        # Rejection Sampling on Fq
        f = np.random.uniform(0.0,fm)
        if Fq(relative_energy(r,vr,vt),r*vt)*vt >= f:
            return vr,vt

print('Sampling velocities...')
ti = time.time()
if use_parallel:
    from multiprocessing import Pool        
    with Pool(nb_threads) as p:
        vels = np.array(p.map(sample_vel,r_rand)).transpose()
else:
    vels = np.empty((2,N))
    for j,r in enumerate(tqdm(r_rand)):
        vels[:,j] = sample_vel(r)
tf = time.time()
print('Sampling took ' + "{:.2f}".format(tf-ti) + ' seconds.')

# Convert to Cartesian
# First: project vt on e_theta, e_phi with random orientation
alph = np.random.uniform(0,2*np.pi,N)
sgn = np.random.choice([-1,1],size=N)
vphi = vels[1]*np.cos(alph)
vtheta = vels[1]*np.sin(alph)
# project vr on e_r (random sign)
vr = sgn*vels[0]

# Convert Spherical to cartesian coordinates
v_x = np.sin(theta_rand)*np.cos(phi_rand)*vr + np.cos(theta_rand)*np.cos(phi_rand)*vtheta - np.sin(phi_rand)*vphi
v_y = np.sin(theta_rand)*np.sin(phi_rand)*vr + np.cos(theta_rand)*np.sin(phi_rand)*vtheta + np.cos(phi_rand)*vphi
v_z = np.cos(theta_rand)*vr - np.sin(theta_rand)*vtheta

# Create velocity array
V = np.array([v_x,v_y,v_z]).transpose()

####### Generate masses

m = M/N * np.ones(N)

####### Exclude extreme outliers
idx = np.sqrt(np.sum(X**2,1)) < bound
X = X[idx]
V = V[idx]
m = m[idx]
new_N = len(m)

###### Write to hdf5
print('Writing IC file...')
with h5py.File(fname,'w') as f:
    wg.write_header(
        f,
        boxsize=box_size,
        flag_entropy = 0,
        np_total = [0,new_N,0,0,0,0],
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
        ids=np.arange(new_N), # Overridden by params.yml
        int_energy = np.zeros(new_N),
        smoothing = np.ones(new_N)
    )

###### Optional: Pickle ic arrays for future use
if pickle_ics:
    import pickle as pkl
    with open('X.pkl','wb') as f:
        pkl.dump(X[:],f)

    with open('V.pkl','wb') as f:
        pkl.dump(V[:],f)

    with open('M.pkl','wb') as f:
        pkl.dump(m[:],f)
