import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm
import argparse
import h5py

parser = argparse.ArgumentParser(description="Plot 2d surface density")

# Snapshot Parameters
parser.add_argument("files",nargs='+',help="snapshot files to be imaged")
parser.add_argument("-shift", type=float, default=1000.0, help="Shift applied to particles in params.yml")
parser.add_argument("-aid",type=int,default=10000000,help="Index below which not to plot particles")

# Histogram Parameters
parser.add_argument("-nbins",type=int,default=500,help="Number of bins in each dimension (x,y)")
parser.add_argument("-lim",type=float,default=14,help="Limits of histogram (in kpc)")
parser.add_argument("-interp",type=str,default='none',help="Interpolation used ('none','kaiser','gaussian',...). Default: none")
parser.add_argument("--polar",action='store_true',help="Pot a r-phi histogram rather than an image")

# Color Normalization Parameters
parser.add_argument("-cmap",type=str,default='YlGnBu_r',help="Colormap (try YlGnBu_r, Magma !)")
parser.add_argument("-norm",type=str,default='log',help='Color Norm to use')
parser.add_argument("-gamma",type=float,help="Exponent of power law norm (use -norm power)")
parser.add_argument("-cmin",type=float,default=-1,help=
                    """Minimum Physical value in the Histogram (is automatically set to >0 for log norm).
                    This effectively sets the contrast of the images. Set to negative to use min value across snapshots""")

# Figure Parameterrs
parser.add_argument("--notex",action='store_true')
parser.add_argument("-nrows",type=int,default=2,help="Number of rows in figure")
parser.add_argument("-ncols",type=int,default=3,help="Number of columns in figure")
parser.add_argument("-figheight",type=float,default=8,help="Height of figure in inches")
parser.add_argument("-figwidth",type=float,default=12,help="Width of figure in inches")
parser.add_argument("--savefig",action='store_true',help="Save figure as .eps")

# Corotation Parameters
parser.add_argument("-rcr",type=float,default=0.0,
    help="Radius of the corotating origin, set to 0 for no corotation (default)")
parser.add_argument("-acr",type=float,default=0,
    help="Initial angle of the corotating origin")
parser.add_argument("-vcr",type=float,default=200,
    help="Circular velocity of the corotation")
parser.add_argument("--set_origin",action='store_true',
    help="Set the origin of the figure to the corotating origin (default: no)")

args = parser.parse_args()
fnames = args.files
if len(fnames) > args.nrows*args.ncols:
    raise ValueError("Too many snapshots to fit in figure")

# Convenience
lim = args.lim
active_id_start = int(args.aid)

# Configure histogram color norm
if args.norm == 'power':
    norm = PowerNorm(gamma=args.gamma)
elif args.norm == 'log':
    # The color range is later automatically adjusted to >0 if log norm
    norm = LogNorm(clip=True)
elif args.norm == 'linear':
    norm = None # Equivalent to linear
else:
    raise NameError("Unknown color norm: " + str(args.norm))

# Configure Pyplot
if not args.notex: plt.rcParams.update({"text.usetex": True,'font.size':1.2*args.figheight,'font.family': 'serif'})
else: plt.rcParams.update({'font.size':15})


# Set up corotation
corot = args.rcr > 0
if corot:
    omega = -args.vcr / args.rcr
    x0, y0 = 0.0,0.0
    if args.set_origin:
        x0 = args.rcr*np.cos(args.acr)
        y0 = args.rcr*np.sin(args.acr)
    def process_pos(X,Y,t):
        x = np.cos(omega*t)*X - np.sin(omega*t)*Y - x0
        y = np.sin(omega*t)*X + np.cos(omega*t)*Y - y0
        return x,y

cmax = 0. # The max of the color range is computed across all snapshots
cmin = 0.
fig, ax = plt.subplots(args.nrows,args.ncols,figsize=(args.figwidth,args.figheight))
ax = ax.flatten()
ims = []
for i,fname in enumerate(fnames):
    # Extract data from snapshot
    f = h5py.File(fname, "r")
    pos = np.array(f["DMParticles"]["Coordinates"]) - args.shift
    IDs = np.array(f["DMParticles"]["ParticleIDs"])
    pos = pos[IDs>=active_id_start]
    t = f["Header"].attrs["Time"][0]
    t_myr = t * f["Units"].attrs["Unit time in cgs (U_t)"][0]/ 31557600.0e6
    x = pos[:,0]
    y = pos[:,1]

    # Process Data
    if corot:
        x,y = process_pos(x,y,t)

    # Make image
    if not args.polar:
        data = np.histogram2d(x,y,bins=args.nbins,range=[[-lim, lim], [-lim, lim]])[0]
        ims.append(ax[i].imshow(data.T, interpolation = args.interp, norm=norm,
                    extent=(-lim,lim,-lim,lim),cmap=args.cmap))
        ax[i].set_xlim(-lim,lim)
        ax[i].set_ylim(-lim,lim)
        ax[i].set_aspect('equal')
        ax[i].set_xlabel('x [kpc]')
        ax[i].set_ylabel('y [kpc]')
    else:
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y,x)
        data,xe,ye,im = ax[i].hist2d(r,phi,range=[[0,lim],[-np.pi,np.pi]],bins=args.nbins,cmap=args.cmap)
        ims.append(im)
        ax[i].set_box_aspect(1)
        ax[i].set_xlabel('$r$ [kpc]')
        ax[i].set_ylabel('$\phi$ [rad]')

    # Find global cmin, cmax
    cmin = min(cmin,np.amin(data))
    cmax = max(cmax,np.amax(data)) 
    ax[i].set_title(r'$t=$ ' + "{:.2f}".format(t_myr) + ' Myr',fontsize=args.figwidth)
    
# Set limits of color range to global min/max across snapshots
if args.cmin > 0.0:
    cmin = args.cmin
print((cmin,cmax))
if cmin <= 0.0 and args.norm == 'log': cmin = 1. # Can be improved...
for im in ims:
    im.set(clim=(cmin,cmax))

# Show & Save figure
plt.show()
if args.savefig:
    fig.savefig('densities_mult.eps',bbox_inches='tight')
