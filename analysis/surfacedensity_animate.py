import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation
from read_snapshot import *
import argparse
import os

parser = argparse.ArgumentParser(description="Plot 2d surface density")

parser.add_argument("dir",type=str,help="Directory of snapshot files")
parser.add_argument("-nbins",type=int,default=200,help="Number of bins in xy")
parser.add_argument("-lim",type=float,default=500,help="Limits of figure (in kpc)")
parser.add_argument("-step",type=float,default=1,help="Step between frames")
parser.add_argument("-interp",type=str,default='none',help="Interpolation used ('none','kaiser','gaussian',...). Default: none")
parser.add_argument("-cmap",type=str,default='YlGnBu_r',help="Colormap")
parser.add_argument("--notex",action='store_true')
parser.add_argument("--savevid",action='store_true')
parser.add_argument("-FPS",type=int,default=20,help="FPS of output mp4")
args = parser.parse_args()

final_output = len(os.listdir(args.dir)) - 2

# Output Parameters
lim = args.lim			# Axes limit (kpc)
figsize = 8			# Figure size
nbins = args.nbins			# Number of xy bins to pass to the raster
shift = 1000.		# Shift applied to the particles (subtracted), see ics
interp = args.interp		# Interpolation to use (none, kaiser, gaussian,..)
norm = LogNorm()	# Norm to use in imshow (linear, log) if log: exclude 0s
cm = args.cmap		# Color map to use (try YlGnBu_r, Magma !)
save_vid = args.savevid
tex = not args.notex
FPS = args.FPS
bitrate = -1 # Lets ffmpeg choose the best bitrate

if tex: plt.rcParams.update({"text.usetex": tex,'font.size':19,'font.family': 'serif'})
else: plt.rcParams.update({'font.size':15})

def update_plot(i):
    fn = args.dir + 'output_' + "{:04n}".format(i) + '.hdf5'
    sn = snapshot(fn)
    x = sn.pos[:,0] - shift
    y = sn.pos[:,1] - shift
    time = sn.time_Myr()
    data,x,y = np.histogram2d(x,y,bins=nbins,range=[[-lim, lim], [-lim, lim]])
    im.set_data(data+1) # +1 if log norm
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)
    ax.set_title(r'$t=$ ' + "{:.2f}".format(time) + ' Myr',fontsize=25)
    #print('### Processing frame ' + str(i) + ' ... ')
    print('Processing ' + fn + ' ...')

# Initialize Plot
fname = args.dir + 'output_0000.hdf5'
sn = snapshot(fname)
x = sn.pos[:,0] - shift
y = sn.pos[:,1] - shift
time = sn.time_Myr()
fig, ax = plt.subplots(figsize=(figsize,figsize))
data, x, y = np.histogram2d(x,y,bins=nbins,range=[[-lim, lim], [-lim, lim]])
im = ax.imshow(data.T+1, interpolation = interp, norm=norm,
			extent=(-lim,lim,-lim,lim),cmap=cm)
ax.set_xlim(-lim,lim)
ax.set_ylim(-lim,lim)
ax.set_xlabel('x [kpc]')
ax.set_ylabel('y [kpc]')
ax.set_title(r'$t=$ ' + "{:.2f}".format(time) + ' Myr',fontsize=25)


# Animate
ani = FuncAnimation(fig, update_plot,frames=range(0,final_output+1,args.step),interval=100, repeat=1)

# Show or save figure
if save_vid:
    ani.save('animation.mp4',writer='ffmpeg',fps=FPS,bitrate=bitrate)
else:
    plt.show()
