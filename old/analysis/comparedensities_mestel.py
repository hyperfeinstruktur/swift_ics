import numpy as np
from matplotlib import pyplot as plt
from read_snapshot import *
import argparse
from scipy.optimize import curve_fit

# Parse user input
parser = argparse.ArgumentParser(description="Plot multiple density profiles against theoretical prediction")
parser.add_argument("files",nargs='+',help="snapshot files to be imaged")
parser.add_argument("--notex",action='store_true',help="Flag to not use LaTeX markup")
parser.add_argument("-r0",type=float,default=10,help="Scale Radius of theoretical Mestel disk")
parser.add_argument("-v0",type=float,default=100,help="Circular velocity of theoretical Mestel disk")
parser.add_argument("-Rcut",type=float,default=160,help="Cut Radius")
parser.add_argument("-chi",type=float,default=0,help="Global Mass Fraction")
parser.add_argument("-aid",type=int,default=0,help="ID below which particles are passive")

args = parser.parse_args()
fnames = args.files

# Limit number of snapshots to plot
if len(fnames) > 20 : raise ValueError("Too many ({:d}) files provided (cannot plot more than 20).".format(len(fnames)))

# Set parameters
tex = not args.notex
if tex: plt.rcParams.update({"text.usetex": tex,'font.size':14,'font.family': 'serif'})
else: plt.rcParams.update({'font.size':12})
shift = 1000.
active_id_start = args.aid
figsize = 7

# Model Parameters (Mestel surface density)
G = 4.299581e+04
rsp = np.logspace(0.0,np.log10(args.Rcut),200)
v0 = args.v0
r0 = args.r0

# Plot densities
fig, ax = plt.subplots(figsize=(figsize,1.2*figsize))
for fname in fnames:
    print(fname)
    # Snapshot
    sn = snapshot(fname)
    idx = sn.IDs>=active_id_start
    pos = sn.pos[idx] - 1000.
    x = pos[:,0]
    y = pos[:,1]
    r = np.sqrt(np.sum(pos**2,1))
    time = sn.time_Myr()
    mass = sn.mass[idx]
    
    # Methods to compute density profile
    def mass_ins(R):
        return ((r<R)*mass).sum()
    
    mass_ins_vect = np.vectorize(mass_ins)
    def density(R):
        return np.diff(mass_ins_vect(R)) / np.diff(R) / (2.*np.pi*R[1:])
    
    def mestel_analytical(r):
        return v0**2/(2.0*np.pi*G*r)
    
    # Plot
    ax.loglog(rsp[1:],density(rsp),'o',ms=1.7,label=r'$t=$ {:.3f} Gyr'.format(sn.time_Gyr()))

#fitstr = 'Mestel disk: $r_0 = {:.1f}$ , $v_0 = {:.1f}$'.format(r0,v0)
fitstr = r'Mestel disk: $v_0 = {:.1f}$ km/s, $\xi = {:.1f}$'.format(v0,args.chi)
ax.plot(rsp,args.chi*mestel_analytical(rsp),c='black',label=fitstr)
#ax.set_xlim(-lim,lim)
#ax.set_ylim(-lim,lim)
ax.set_xlabel('r [kpc]')
ax.legend()
plt.tight_layout()
ax.set_ylabel(r'$\Sigma(r)$ [$M_{\odot}$ kpc$^{-2}$]')
plt.show()
