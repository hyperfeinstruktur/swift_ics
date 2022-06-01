import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm
from read_snapshot import *
import argparse

parser = argparse.ArgumentParser(description="Plot 2d surface density")

parser.add_argument("file",type=str,help="Name of the snapshot file to be imaged")
parser.add_argument("-nbins",type=int,default=400,help="Number of bins in xy")
parser.add_argument("-lim",type=float,default=100,help="Limits of figure (in kpc)")
parser.add_argument("-cmap",type=str,default='YlGnBu_r',help="Colormap")
parser.add_argument("-interp",type=str,default='none',help="Interpolation used ('none','kaiser','gaussian',...). Default: none")
parser.add_argument("--notex",action='store_true')
parser.add_argument("--PowerNorm",action='store_true')
parser.add_argument("-gamma",type=float,help="Exponent of power law norm (use --PowerNorm)")
args = parser.parse_args()
fname = str(args.file)

# Output Parameters
lim = args.lim			# Axes limit (kpc)
figsize = 8			# Figure size
nbins = args.nbins			# Number of xy bins to pass to the raster
shift = 1000.		# Shift applied to the particles (subtracted), see ics
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
active_id_start = 1e7

sn = snapshot(fname)
pos = sn.pos[sn.IDs>=active_id_start]
x = pos[:,0] - shift
y = pos[:,1] - shift
time = sn.time_Myr()
fig, ax = plt.subplots(figsize=(figsize,figsize))
data, x, y = np.histogram2d(x,y,bins=nbins,range=[[-lim, lim], [-lim, lim]])
im = ax.imshow(data.T+1, interpolation = interp, norm=norm,
			extent=(-lim,lim,-lim,lim),cmap=cm)
ax.set_xlim(-lim,lim)
ax.set_ylim(-lim,lim)
ax.set_xlabel('x [kpc]')
ax.set_ylabel('y [kpc]')
ax.set_title(r'$t=$' + "{:.2f}".format(time) + ' Myr',fontsize=25)
plt.show()
