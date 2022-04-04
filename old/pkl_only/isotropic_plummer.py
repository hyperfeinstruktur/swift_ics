import numpy as np
from matplotlib import pyplot as plt
import pickle as pkl

####### Model Parameters

G = 4.299581e+04  # Gravitational constant [kpc / 10^10 M_s * (kms)^2]
a = 0.01          # Plummer softening length [kpc]
M = 6.0e-6        # Total Mass [10^10 M_s]
N = 10000        # Number of Particles
bound = 2.0		  # Max distance to origin (exclude outliers)

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
for i in range(len(vel_rand)):
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

##### Exclude extreme outliers
idx = np.sqrt(np.sum(X**2,1)) < bound
X = X[idx]
V = V[idx]
m = m[idx]

# Pickle results
if 1:
    with open('X.pkl','wb') as f:
        pkl.dump(X[:],f)

    with open('V.pkl','wb') as f:
        pkl.dump(V[:],f)
        #pkl.dump(np.zeros(V.shape),f)

    with open('M.pkl','wb') as f:
        pkl.dump(m[:],f)


#plt.scatter(x,y,s=0.4)
#plt.show()

# Dynamical Time
pc =  3.0857e13 # km
print('Dynamical Time:')
vm = np.sqrt(np.mean((V**2).sum(axis=1)))
print(str(2*a*pc/vm / 86400.0 / 1e9) + ' Gyrs')
