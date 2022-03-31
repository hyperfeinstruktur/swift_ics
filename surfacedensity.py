import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from read_snapshot import *
import argparse

parser = argparse.ArgumentParser(description="Plot 2d surface density")

parser.add_argument("file",type=str,help="Name of the snapshot file to be imaged")
parser.add_argument("-nbins",type=int,default=200,help="Number of bins in xy")
parser.add_argument("-lim",type=float,default=0.07,help="Limits of figure (in kpc)")
parser.add_argument("-interp",type=str,default='none',help="Interpolation used ('none','kaiser','gaussian',...). Default: none")
parser.add_argument("--notex",action='store_true')
args = parser.parse_args()
fname = str(args.file)

# Output Parameters
lim = args.lim			# Axes limit (kpc)
figsize = 8			# Figure size
nbins = args.nbins			# Number of xy bins to pass to the raster
shift = 2.		# Shift applied to the particles (subtracted), see ics
interp = args.interp		# Interpolation to use (none, kaiser, gaussian,..)
norm = LogNorm()	# Norm to use in imshow (linear, log) if log: exclude 0s
cm = 'YlGnBu_r'		# Color map to use (try YlGnBu_r, Magma !)
tex = not args.notex
if tex: plt.rcParams.update({"text.usetex": tex,'font.size':19,'font.family': 'serif'})
else: plt.rcParams.update({'font.size':15})

sn = snapshot(fname)
x = sn.pos[:,0] - shift
y = sn.pos[:,1] - shift
time = sn.time_Gyr()
fig, ax = plt.subplots(figsize=(figsize,figsize))
data, x, y = np.histogram2d(x,y,bins=nbins,range=[[-lim, lim], [-lim, lim]])
im = ax.imshow(data.T+1, interpolation = interp, norm=norm,
			extent=(-lim,lim,-lim,lim),cmap=cm)
ax.set_xlim(-lim,lim)
ax.set_ylim(-lim,lim)
ax.set_xlabel('x [kpc]')
ax.set_ylabel('y [kpc]')
ax.set_title(r'$t=$' + "{:.2f}".format(time) + 'Gyr',fontsize=25)
plt.show()
