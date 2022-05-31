import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm
import argparse
import h5py

parser = argparse.ArgumentParser(description="Plot 2d surface density")

parser.add_argument("files",nargs='+',help="snapshot files to be imaged")
parser.add_argument("-nbins",type=int,default=800,help="Number of bins in xy")
parser.add_argument("-lim",type=float,default=26,help="Limits of figure (in kpc)")
parser.add_argument("-cmap",type=str,default='YlGnBu_r',help="Colormap")
parser.add_argument("-interp",type=str,default='none',help="Interpolation used ('none','kaiser','gaussian',...). Default: none")
parser.add_argument("--notex",action='store_true')
parser.add_argument("--PowerNorm",action='store_true')
parser.add_argument("-aid",type=int,default=10000000)
parser.add_argument("-gamma",type=float,help="Exponent of power law norm (use --PowerNorm)")
parser.add_argument(
    "-shift", type=float, default=1000.0, help="Shift applied to particles in params.yml"
)
parser.add_argument("-savefig",action='store_true')
parser.add_argument("-nrows",type=int,default=2,help="Number of rows in figure")
parser.add_argument("-ncols",type=int,default=3,help="Number of rows in figure")
parser.add_argument("-figheight",type=float,default=8,help="Height of figure in inches")
parser.add_argument("-figwidth",type=float,default=12,help="Width of figure in inches")

# Parameters to follow the perturbation (rotating frame)
parser.add_argument("-rp",type=float,default=0.0,
    help="Radius of the corotating origin (set to 0 for no corotation")
parser.add_argument("-ap",type=float,default=0,
    help="Initial angle of the corotating origin")
parser.add_argument("-vp",type=float,default=200,
    help="Circular velocity of the perturber")

args = parser.parse_args()
fnames = args.files
if len(fnames) > args.nrows*args.ncols:
    raise ValueError("Too many snapshots to fit in figure")

# Output Parameters
lim = args.lim			# Axes limit (kpc)
nbins = args.nbins			# Number of xy bins to pass to the raster
interp = args.interp		# Interpolation to use (none, kaiser, gaussian,..)
if args.PowerNorm:
    norm = PowerNorm(gamma=args.gamma)#LogNorm()	# Norm to use in imshow (linear, log) if log: exclude 0s
else:
    norm = LogNorm()
cm = args.cmap		# Color map to use (try YlGnBu_r, Magma !)
tex = not args.notex
if tex: plt.rcParams.update({"text.usetex": tex,'font.size':args.figheight,'font.family': 'serif'})
else: plt.rcParams.update({'font.size':15})

# ID below which not to plot particles (unresponsive ones), leave at 0 if none
active_id_start = int(args.aid)

# Rotating frame
if args.rp > 0:
    omega = -args.vp / args.rp      # Angular velocity of the perturber
    x0 = args.rp*np.cos(args.ap)    # Initial position of the perturber
    y0 = args.rp*np.sin(args.ap)
else:
    omega = 0.0
    x0 = 0.0
    y0 = 0.0

fig, ax = plt.subplots(args.nrows,args.ncols,figsize=(args.figwidth,args.figheight))
ax = ax.flatten()
for i,fname in enumerate(fnames):
    f = h5py.File(fname, "r")
    pos = np.array(f["DMParticles"]["Coordinates"]) - args.shift
    IDs = np.array(f["DMParticles"]["ParticleIDs"])
    pos = pos[IDs>=active_id_start]
    t = f["Header"].attrs["Time"][0]
    time = (
        f["Header"].attrs["Time"][0]
        * f["Units"].attrs["Unit time in cgs (U_t)"][0]
        / 31557600.0e6
    )
    X = pos[:,0]
    Y = pos[:,1]
    # Rotate Frame
    x = np.cos(omega*t)*X - np.sin(omega*t)*Y # - x0
    y = np.sin(omega*t)*X + np.cos(omega*t)*Y # - y0

    # Plot data
    #data = np.histogram2d(x,y,bins=nbins,range=[[-lim, lim], [-lim, lim]])[0]
    #im = ax[i].imshow(data.T+1, interpolation = interp, norm=norm,
    #			extent=(-lim,lim,-lim,lim),cmap=cm)
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y,x)
    ax[i].hist2d(r,phi,range=[[0,lim],[-np.pi,np.pi]],bins=nbins,cmap=cm)
    #ax.hist2d(x,y,bins=nbins,range=[[-lim, lim], [-lim, lim]],cmap=cm,norm=norm)
    #ax[i].set_xlim(-lim,lim)
    #ax[i].set_ylim(-lim,lim)
    ax[i].set_xlabel('$r$ [kpc]')
    ax[i].set_ylabel('$\phi$ [rad]')
    ax[i].set_title(r'$t=$ ' + "{:.2f}".format(time) + ' Myr',fontsize=args.figwidth)
    #ax[i].set_aspect('equal')
plt.show()
if args.savefig:
    fig.savefig('densities6.eps',bbox_inches='tight')
