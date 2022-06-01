import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm
import argparse
import h5py

parser = argparse.ArgumentParser(description="Plot 2d surface density")

parser.add_argument("file",type=str,help="Name of the snapshot file to be imaged")
parser.add_argument("-nbins",type=int,default=400,help="Number of bins in xy")
parser.add_argument("-lim",type=float,default=100,help="Limits of figure (in kpc)")
parser.add_argument("-cmap",type=str,default='YlGnBu_r',help="Colormap")
parser.add_argument("-interp",type=str,default='none',help="Interpolation used ('none','kaiser','gaussian',...). Default: none")
parser.add_argument("--notex",action='store_true')
parser.add_argument("--PowerNorm",action='store_true')
parser.add_argument("-aid",type=int,default=10000000)
parser.add_argument("-gamma",type=float,help="Exponent of power law norm (use --PowerNorm)")
parser.add_argument(
    "-shift", type=float, default=1000.0, help="Shift applied to particles in params.yml"
)
args = parser.parse_args()
fname = str(args.file)

# Output Parameters
lim = args.lim			# Axes limit (kpc)
figsize = 8			# Figure size
nbins = args.nbins			# Number of xy bins to pass to the raster
interp = args.interp		# Interpolation to use (none, kaiser, gaussian,..)
if args.PowerNorm:
    norm = PowerNorm(gamma=args.gamma)#LogNorm()	# Norm to use in imshow (linear, log) if log: exclude 0s
else:
    norm = LogNorm()
cm = args.cmap		# Color map to use (try YlGnBu_r, Magma !)
tex = not args.notex
if tex: plt.rcParams.update({"text.usetex": tex,'font.size':19,'font.family': 'serif'})
else: plt.rcParams.update({'font.size':15})

# ID below which not to plot particles (unresponsive ones), leave at 0 if none
active_id_start = int(args.aid)

f = h5py.File(fname, "r")
pos = np.array(f["DMParticles"]["Coordinates"]) - args.shift
IDs = np.array(f["DMParticles"]["ParticleIDs"])
pos = pos[IDs>=active_id_start]
time = (
    f["Header"].attrs["Time"][0]
    * f["Units"].attrs["Unit time in cgs (U_t)"][0]
    / 31557600.0e6
)
x = pos[:,0]
y = pos[:,1]
fig, ax = plt.subplots(figsize=(figsize,figsize))
data, x, y = np.histogram2d(x,y,bins=nbins,range=[[-lim, lim], [-lim, lim]])
im = ax.imshow(data.T+1, interpolation = interp, norm=norm,
			extent=(-lim,lim,-lim,lim),cmap=cm)
#ax.hist2d(x,y,bins=nbins,range=[[-lim, lim], [-lim, lim]],cmap=cm,norm=norm)
ax.set_xlim(-lim,lim)
ax.set_ylim(-lim,lim)
ax.set_xlabel('x [kpc]')
ax.set_ylabel('y [kpc]')
ax.set_title(r'$t=$ ' + "{:.2f}".format(time) + ' Myr',fontsize=25)
plt.show()
