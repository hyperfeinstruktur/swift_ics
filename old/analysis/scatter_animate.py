import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation
from read_snapshot import *
import argparse
import os

parser = argparse.ArgumentParser(description="Animated plot of particle positions")

parser.add_argument("dir",type=str,help="Directory of snapshot files")
parser.add_argument("-lim",type=float,default=500,help="Limits of figure (in kpc)")
parser.add_argument("-step",type=float,default=1,help="Step between frames")
parser.add_argument("--notex",action='store_true')
parser.add_argument("--savevid",action='store_true')
parser.add_argument("-dotsize",type=float,default=0.5,help="Size of Scatter dots")
args = parser.parse_args()

final_output = len(os.listdir(args.dir)) - 2

# Output Parameters
lim = args.lim			# Axes limit (kpc)
figsize = 8			# Figure size
shift = 1000.		# Shift applied to the particles (subtracted), see ics
dotsize = args.dotsize
save_vid = args.savevid
tex = not args.notex

if tex: plt.rcParams.update({"text.usetex": tex,'font.size':19,'font.family': 'serif'})
else: plt.rcParams.update({'font.size':15})

def update_plot(i):
    fn = args.dir + 'output_' + "{:04n}".format(i) + '.hdf5'
    print(fn)
    sn = snapshot(fn)
    time = sn.time_Myr()
    sc.set_offsets(sn.pos[:,0:2]-shift)
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)
    ax.set_title(r'$t=$' + "{:.2f}".format(time) + ' Myr',fontsize=25)
    #print('### Processing frame ' + str(i) + ' ... ')
    print('Processing ' + fn + ' ...')

fname = args.dir + 'output_0000.hdf5'
sn = snapshot(fname)
x = sn.pos[:,0] - shift
y = sn.pos[:,1] - shift
time = sn.time_Myr()
fig, ax = plt.subplots(figsize=(figsize,figsize))
sc = ax.scatter(x,y,s=dotsize,c='white')
ax.set_facecolor('black')
ax.set_xlim(-lim,lim)
ax.set_ylim(-lim,lim)
ax.set_xlabel('x [kpc]')
ax.set_ylabel('y [kpc]')
ax.set_aspect('equal','box')
ax.set_title(r'$t=$' + "{:.2f}".format(time) + ' Myr',fontsize=25)


# Animate
ani = FuncAnimation(fig, update_plot,frames=range(0,final_output+1,args.step), interval=100, repeat=1)

# Show or save figure
if save_vid:
    ani.save('anim_dens_70pc_400bins_log.mp4',writer='ffmpeg',fps=FPS,bitrate=bitrate)
else:
    plt.show()
