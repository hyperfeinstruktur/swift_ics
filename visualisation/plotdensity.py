import numpy as np
from matplotlib import pyplot as plt
import h5py
import argparse
from scipy.optimize import curve_fit

# Parse user input
parser = argparse.ArgumentParser(description="Plot multiple density profiles against theoretical prediction")
parser.add_argument("files",nargs='+',help="snapshot files to be imaged")
parser.add_argument("--notex",action='store_true',help="Flag to not use LaTeX markup")
parser.add_argument("-v0",type=float,default=200,help="Circular velocity of theoretical Mestel disk")
parser.add_argument("-rmin",type=float,default=0.5,help="Lower bound of radius to plot")
parser.add_argument("-rmax",type=float,default=20,help="Upper bound of radius to plot")
parser.add_argument("-chi",type=float,default=0,help="Global Mass Fraction in particles (1-chi in ext potential)")
parser.add_argument("-aid",type=int,default=0,help="ID below which particles are passive")
parser.add_argument("-shift", type=float, default=1000.0, help="Shift applied to particles in params.yml")

args = parser.parse_args()
fnames = args.files

# Limit number of snapshots to plot
if len(fnames) > 20 : raise ValueError("Too many ({:d}) files provided (cannot plot more than 20).".format(len(fnames)))

# Set parameters
tex = not args.notex
if tex: plt.rcParams.update({"text.usetex": tex,'font.size':16,'font.family': 'serif'})
else: plt.rcParams.update({'font.size':12})
active_id_start = args.aid
figsize = 7

# Model Parameters (Mestel surface density)
G = 4.299581e+04
rsp = np.logspace(np.log10(args.rmin),np.log10(args.rmax),200)
v0 = args.v0

# Plot densities
fig, ax = plt.subplots(figsize=(figsize,figsize))
for fname in fnames:
    print(fname)
    # Snapshot
    f = h5py.File(fname, "r")
    IDs = np.array(f["DMParticles"]["ParticleIDs"])
    # Read positions & masses and select "active" particles (if desired)
    pos = np.array(f["DMParticles"]["Coordinates"]) - args.shift
    pos = pos[IDs>=active_id_start]
    mass = np.array(f["DMParticles"]["Masses"])
    mass = mass[IDs>=active_id_start]
    # Read time in Myrs (for display)
    time = (
    f["Header"].attrs["Time"][0]
    * f["Units"].attrs["Unit time in cgs (U_t)"][0]
    / 31557600.0e6
    )
    # Process snapshot data
    x = pos[:,0]
    y = pos[:,1]
    r = np.sqrt(np.sum(pos**2,1))
    
    # Methods to compute density profile
    def mass_ins(R):
        return ((r<R)*mass).sum()
    
    mass_ins_vect = np.vectorize(mass_ins)
    def density(R):
        return np.diff(mass_ins_vect(R)) / np.diff(R) / (2.*np.pi*R[1:])
    
    def mestel_analytical(r):
        return v0**2/(2.0*np.pi*G*r)
    
    # Plot
    ax.loglog(rsp[1:],density(rsp),'o',ms=1.7,label=r'$t=$ {:.3f} Gyr'.format(time))

fitstr = r'Mestel disk: $v_0 = {:.1f}$ km/s, $\xi = {:.1f}$'.format(v0,args.chi)
ax.plot(rsp,args.chi*mestel_analytical(rsp),c='black',label=fitstr)
#ax.set_xlim(-lim,lim)
#ax.set_ylim(-lim,lim)
ax.set_xlabel('r [kpc]',fontsize=20)
ax.set_ylabel(r'$\Sigma(r)$ [$M_{\odot}$ kpc$^{-2}$]',fontsize=20)
ax.legend(fontsize=14)
plt.tight_layout()
plt.show()
