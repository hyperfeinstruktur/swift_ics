from pNbody import *
import pickle as pkl

# Import ICs
with open('X.pkl','rb') as f:
    x = pkl.load(f)
with open('V.pkl','rb') as f:
    v = pkl.load(f)
with open('M.pkl','rb') as f:
    m = pkl.load(f)

nb = Nbody(
status='new',
p_name='plummer_test.hdf5',
pos=x + 2.0, # +2kpc shift
vel=v,
mass=m,
ftype='swift')

nb.set_tpe(1)
nb.boxsize = np.array([4.,4.,4.])
nb.flag_entr_ic = 0.0 # A mentionner prochain rdv

nb.UnitLength_in_cm = 3.086e21 # 1kpc
nb.UnitVelocity_in_cm_per_s = 1e5 # 1km/s
nb.UnitMass_in_g = 1.988e43 # 10^10 Solar masses

nb.info()
timeUnit = nb.UnitLength_in_cm / nb.UnitVelocity_in_cm_per_s
print('Time Unit: ' + str(timeUnit / 86400 / 1e9) + ' Gyrs')
vmean = np.mean(np.sqrt(np.sum(v**2,axis=1))) * nb.UnitVelocity_in_cm_per_s
dyntime = 0.04*nb.UnitLength_in_cm /vmean / timeUnit
print('Dynamical Time (in internal units) : ' + str(dyntime))
nb.write()
