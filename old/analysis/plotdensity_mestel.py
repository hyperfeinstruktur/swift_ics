import numpy as np
from matplotlib import pyplot as plt
from read_snapshot import *
import argparse
from scipy.optimize import curve_fit

parser = argparse.ArgumentParser(description="Check the density generated by the particles")
parser.add_argument("file",type=str,help="Name of the snapshot file to be imaged")
parser.add_argument("--notex",action='store_true')
parser.add_argument("-r0",type=float,default=10,help="Scale Radius initial guess")
parser.add_argument("-v0",type=float,default=100,help="Circular velocity initial guess")
parser.add_argument("-chi",type=float,default=0,help="Global Mass Fraction")
parser.add_argument("-aid",type=int,default=0,help="ID below which particles are passive")

args = parser.parse_args()
fname = str(args.file)

tex = not args.notex
if tex: plt.rcParams.update({"text.usetex": tex,'font.size':14,'font.family': 'serif'})
else: plt.rcParams.update({'font.size':12})
shift = 1000.
active_id_start = args.aid
figsize = 6

# Snapshot
sn = snapshot(fname)
idx = sn.IDs>=active_id_start
pos = sn.pos[idx] - 1000.
x = pos[:,0]
y = pos[:,1]
r = np.sqrt(np.sum(pos**2,1))
time = sn.time_Myr()
mass = sn.mass[idx]

# Model Parameters
G = 4.299581e+04
rsp = np.logspace(0.0,np.log10(160),200)
v0 = args.v0
r0 = args.r0

# Methods to compute density profile
def mass_ins(R):
    return ((r<R)*mass).sum()

mass_ins_vect = np.vectorize(mass_ins)
def density(R):
    return np.diff(mass_ins_vect(R)) / np.diff(R) / (2.*np.pi*R[1:])

def mestel_analytical(r):
    return v0**2/(2.0*np.pi*G*r)

# Plot
fig, ax = plt.subplots(figsize=(figsize,figsize))
ax.loglog(rsp[1:],density(rsp)/(1.0-args.chi),'o',ms=3,label='Snapshot')
fitstr = 'Mestel disk: $r_0 = {:.1f}$ , $v_0 = {:.1f}$'.format(r0,v0)
ax.plot(rsp,mestel_analytical(rsp),c='black',label=fitstr)
#ax.set_xlim(-lim,lim)
#ax.set_ylim(-lim,lim)
ax.set_xlabel('r [kpc]')
ax.legend()
plt.tight_layout()
ax.set_ylabel(r'$\rho(r)$ [$M_{\odot}$ kpc$^{-3}$]')
#ax.set_title(r'$t=$' + "{:.2f}".format(time) + ' Myr')
plt.show()
