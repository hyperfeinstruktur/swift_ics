import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from read_snapshot import *
import argparse

parser = argparse.ArgumentParser(description="Scatter particle position")

parser.add_argument("file",type=str,help="Name of the snapshot file to be imaged")
parser.add_argument("-lim",type=float,default=500,help="Limits of figure (in kpc)")
parser.add_argument("-dotsize",type=float,default=0.5,help="Size of Scatter dots")
parser.add_argument("--notex",action='store_true')
args = parser.parse_args()
fname = str(args.file)

# Output Parameters
lim = args.lim			# Axes limit (kpc)
figsize = 8			# Figure size
shift = 1000.		# Shift applied to the particles (subtracted), see ics
dotsize = args.dotsize
tex = not args.notex
if tex: plt.rcParams.update({"text.usetex": tex,'font.size':19,'font.family': 'serif'})
else: plt.rcParams.update({'font.size':15})

sn = snapshot(fname)
x = sn.pos[:,0] - shift
y = sn.pos[:,1] - shift
time = sn.time_Myr()
fig, ax = plt.subplots(figsize=(figsize,figsize))
ax.scatter(x,y,s=dotsize,c='white')
ax.set_facecolor('black')
ax.set_aspect('equal', 'box')
ax.set_xlim(-lim,lim)
ax.set_ylim(-lim,lim)
ax.set_xlabel('x [kpc]')
ax.set_ylabel('y [kpc]')
ax.set_title(r'$t=$' + "{:.2f}".format(time) + ' Myr',fontsize=25)
plt.show()
