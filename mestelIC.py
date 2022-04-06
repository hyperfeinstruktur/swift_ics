import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sci
from scipy.optimize import minimize
from tqdm import tqdm
import write_gadget as wg
import h5py
import time

if __name__ == '__main__':
    ##### Model Parameters
    # Mestel Disk (Mestel 1963) + Naive cut function
    G = 4.299581e+04 # kpc / 10^10 solar mass * (km/s)^2
    r0 = 10.    # Scale Radius
    q = 11.4   # Dispersion parameter
    v0 = 230  # Circular velocity
    M_cut = 1.0e2
    M_cut_inner = 5.0

    # IC File
    N = 100000       # Number of Particles
    bound = 1000.		  # Max distance to origin (exclude outliers)
    fname = 'mestel.hdf5' # Name of the ic file (dont forget .hdf5)
    pickle_ics = 0      # Optional: Pickle ics (pos,vel,mass) for future use
    box_size = 2000.0
    periodic_bdry = 0

    # Parallelism for velocity sampling
    use_parallel = 1
    nb_threads = None  # Set to None to use os.cpucount()

    # Compute other quantities from parameters
    Sigma0 = v0**2 / (2.0*np.pi*G*r0) # Scale Surface density 
    sig = v0 / np.sqrt(1.0+q) # Velocity dispersion
    F0 = Sigma0 / ( 2.0**(q/2.) * np.sqrt(np.pi) * r0**q * sig**(q+2.0) + sci.gamma((1+q)/2.0)) # DF Normalization factor

    ###### Generate Positions
    def r_of_m(m):
        return G*m/(v0**2)
    R_cut = r_of_m(M_cut)
    R_cut_inner = r_of_m(M_cut_inner)

    m_rand = np.random.uniform(M_cut_inner,M_cut,N)
    r_rand = r_of_m(m_rand)
    theta_rand = np.random.uniform(0.0,2*np.pi,N)

    x = r_rand*np.cos(theta_rand)
    y = r_rand*np.sin(theta_rand)
    z = np.zeros(N)

    X = np.array([x,y,z]).transpose()

    ######## Generate velocities
    # Helper Functions
    def relative_potential(r):
        return -v0**2 * np.log(r/r0)
    def relative_energy(r,vr,vt):
        return relative_potential(r) - 0.5*(vr**2 + vt**2)

    # Base Distribution Function (Toomre 1977)
    def F_M(E,L):
        if L < 0.0 : return 0.0
        else: return F0*L**q*np.exp(E/(sig**2))

    def F_Mv(r,vr,vt):
        return F_M(relative_energy(r,vr,vt),r*vt)

    def sample_vel(r):
        # Compute max velocity (equiv. condition for E>=0)
        #vmax = np.sqrt(2. * relative_potential(r))
        vmax = 3*v0
        # Compute max of DF at this radius
        fm = F_Mv(r,0.0,sig*np.sqrt(q))
        #print(fm)
        while True:
            # Select candidates for vr,vt based on max full velocity
            vr = np.random.uniform(0.0,vmax)
            vt = np.random.uniform(0.0,vmax)
            # Rejection Sampling on Fq
            f = np.random.uniform(0.0,fm)
            if F_Mv(r,vr,vt) >= f:
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

    # Convert polar to cartesian
    sgn = np.random.choice([-1,1],size=N)
    vr = sgn*vels[0]
    vtheta = vels[1]

    v_x = np.cos(theta_rand) * vr - np.sin(theta_rand) * vtheta
    v_y = np.sin(theta_rand) * vr + np.cos(theta_rand) * vtheta
    v_z = np.zeros(N)

    # Create velocity array
    V = np.array([v_x,v_y,v_z]).transpose()

    ####### Generate masses

    m = (M_cut-M_cut_inner)/N * np.ones(N)

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
    print('done.')
    print('Generated Truncated Mestel disk with cuts:')
    print('M_inner = ' + "{:.1e}".format(M_cut_inner) + '   -->   R_inner = ' + "{:.1e}".format(R_cut_inner))
    print('M_outer = ' + "{:.1e}".format(M_cut) + '   -->   R_outer = ' + "{:.1e}".format(R_cut))
    epsilon0 = np.sqrt(2.0*G*(M_cut - M_cut_inner)*r0)/v0 / np.sqrt(N)
    print('Recommended Softening length (times conventional factor): ' + "{:.4e}".format(epsilon0) + ' kpc')
    ###### Optional: Pickle ic arrays for future use
    if pickle_ics:
        import pickle as pkl
        with open('X.pkl','wb') as f:
            pkl.dump(X[:],f)

        with open('V.pkl','wb') as f:
            pkl.dump(V[:],f)

        with open('M.pkl','wb') as f:
            pkl.dump(m[:],f)
