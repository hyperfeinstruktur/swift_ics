import numpy as np
from tqdm import tqdm
import h5py
import write_gadget as wg

####### Parameters
# Plummer Model
G = 4.299581e+04  # Gravitational constant [kpc / 10^10 M_s * (kms)^2]
a = 0.01          # Plummer softening length [kpc]
M = 6.0e-6        # Total Mass [10^10 M_s]

# IC File
N = 1000        # Number of Particles
bound = 2.0		  # Max distance to origin (exclude outliers)
fname = 'plummer_isotropic.hdf5' # Name of the ic file (dont forget .hdf5)
pickle_ics = 0      # Optional: Pickle ics (pos,vel,mass) for future use
box_size = 4.0
periodic_bdry = 0

##### Generate Positions
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
print(np.amax(np.sqrt(np.sum(X**2,1))))

###### Generate Velocities
def relative_potential(r,a,M): # = - phi
    return G*M / np.sqrt(r**2 + a**2)

def relative_energy(r,v,a,M):
    return relative_potential(r,a,M) - 0.5 * v**2

def DF(r,v,a,M):
    A = 24. * np.sqrt(2) / (7*np.pi**3)
    eps = relative_energy(r,v,a,M)
    if eps < 0: return 0
    else : return A * G**(-5.) * M**(-4.) * a**2 * eps**(3.5)

def v_max(r,a,M):
    return np.sqrt(2.*relative_potential(r,a,M))

def sample_velocity(r,a,M):
    v_e = v_max(r,a,M)
    # x = 0.
    # y = 0.
    # while True:
    #     x = v_e*np.random.uniform(0.,1.)
    #     y = DF(r,0.,a,M)*np.random.uniform(0.,1.) # The DF is maximal at r=0 (analytic)
    #     if y < DF(r,x,a,M) and x < v_max(r,a,M): return x,y
    while True:
        X4 = np.random.uniform()
        X5 = np.random.uniform()
        if 0.1*X5 < (X4**2*(1.-X4**2)**(3.5)): return X4*v_e, X5*v_e

vel_rand = np.empty(N)
print('Sampling Velocities...')
for i in tqdm(range(len(vel_rand))):
    vel_rand[i] = sample_velocity(r_rand[i],a,M)[0]

phi_rand = np.random.uniform(0.0,2*np.pi,N)
theta_rand = np.arccos( np.random.uniform(-1.,1.,N) )

v_x = vel_rand * np.sin(theta_rand) * np.cos(phi_rand)
v_y = vel_rand * np.sin(theta_rand) * np.sin(phi_rand)
v_z = vel_rand * np.cos(theta_rand)

V = np.array([v_x,v_y,v_z]).transpose()
print(np.amax(V))

##### Generate masses
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
