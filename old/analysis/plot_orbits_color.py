import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import argparse
plt.rcParams.update({"text.usetex": True,'font.size':20,'font.family': 'serif'})

# Parse Parameters
parser = argparse.ArgumentParser(description="Plot Orbits in a corotating frame")

parser.add_argument("files",nargs='+',
    help="snapshot files to be imaged")
parser.add_argument("-rp",type=float,default=0.0,
    help="Radius of the corotating origin (set to 0 for no corotation")
parser.add_argument("-ap",type=float,default=0,
    help="Initial angle of the corotating origin")
parser.add_argument("-v0",type=float,default=100,
    help="Circular velocity of the model")
parser.add_argument("-rlim",type=float,default=30,
    help="Width of the range of radii (+/- rp) to include")
parser.add_argument("-plotlim",type=float,
    help="Uniform limit around corotating origin to plot")
parser.add_argument("-lw",type=float,default=0.8,
    help="Line Width of orbits on plot")
parser.add_argument("-subsample_prob",type=float,default=1,
    help="Parameter to reduce the number of particles to plot randomly with prob")
parser.add_argument("--mark_last",action='store_true',
    help="Put a Marker at the final position of bodies")

args = parser.parse_args()
fnames = args.files
shift = 1000.

# Global Constants & Parameters
if args.rp > 0:
    omega = -args.v0 / args.rp      # Angular velocity of the perturber
    x0 = args.rp*np.cos(args.ap)    # Initial position of the perturber
    y0 = args.rp*np.sin(args.ap)
else:
    omega = 0.0
    x0 = 0.0
    y0 = 0.0

# Select Orbits to track
"""
First, select according to radius range (rp +/- rlim)
Then, apply a random mask that excludes on average a fraction
[1-subsample_prob] of the remaining orbits.
"""
f = h5py.File(fnames[0],'r')
pos = np.array(f['DMParticles']['Coordinates']) - shift
IDs = np.array(f['DMParticles']['ParticleIDs'])
r = np.sqrt(np.sum(pos**2,1))
idx = np.logical_and(r <= args.rp+args.rlim , r>= args.rp-args.rlim)
idx = np.logical_and(idx, np.random.choice(
    [0,1],size=len(idx),p=[1-args.subsample_prob,args.subsample_prob])
    )
nb_tracked = idx.sum()
track_IDs = IDs[idx]

# Arrays to contain data
output = np.empty((len(fnames),2,nb_tracked))
times = np.empty((len(fnames)))

# Extract Orbits & corresp. times (converted to Gyr)
for i,fn in enumerate(fnames):
    f = h5py.File(fn,'r')
    IDs = np.array(f['DMParticles']['ParticleIDs'])
    idd = np.intersect1d(track_IDs,IDs,return_indices=True)[2]
    pos = np.array(f['DMParticles']['Coordinates'])[idd]
    t = (
        f['Header'].attrs['Time'][0]* f["Units"].attrs["Unit time in cgs (U_t)"][0]
        / 31557600.0e9
    )
    # Raw orbits
    X = pos[:,0] - shift
    Y = pos[:,1] - shift

    # Switch to corotating frame (Flip x/y for display purposes)
    output[i][1] = np.cos(omega*t)*X - np.sin(omega*t)*Y - x0
    output[i][0] = np.sin(omega*t)*X + np.cos(omega*t)*Y - y0
    times[i] = t

# Figure Parameters
lim = args.plotlim
ms = 1
fgs = 7

# Figure init
fig, ax = plt.subplots(figsize=(1.25*fgs,fgs))
if lim is not None:
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)
else:
    ax.set_xlim(1.1*np.amin(output[:,0,:]),1.1*np.amax(output[:,0,:]))
    ax.set_ylim(1.1*np.amin(output[:,1,:]),1.1*np.amax(output[:,1,:]))
ax.set_aspect('equal')
ax.set_xlabel('x [kpc]')
ax.set_ylabel('y [kpc]')

## Plot orbits coloured according to time:
## Following https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html

# Reshape Data
norm = plt.Normalize(times.min(),times.max())
x = output[:,0,:]
y = output[:,1,:]

# Plot Orbits sequencially with colour ~ time
for j in range(nb_tracked):
    segs = np.array([x[:-1,j], y[:-1,j], x[1:,j], y[1:,j]]).T.reshape(-1, 2, 2)
    lc = LineCollection(segs,norm=norm,cmap='Blues',array=times,linewidth=args.lw)
    lines = ax.add_collection(lc)

# Since times are the same for each orbit, colorbar the last "lines" object is fine
fig.colorbar(lines,ax=ax,label=r'Time [Gyr]')

# Add scatter point at final position (optional)
if args.mark_last:
    ax.scatter(x[-1,:],y[-1,:],s=ms,c='black')

plt.title("Orbits in perturbed Mestel disk (Corotating)")
plt.show()