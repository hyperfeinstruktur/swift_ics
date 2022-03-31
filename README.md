# SWIFT N-body isolated stellar ICs
A set of python scripts to generate realizations of a few isolated stellar systems via their distribution function, and to write them to an initial conditions file compatible with the SWIFT code: https://swift.dur.ac.uk/
- plummerIC.py: Generates realization of general Plummer model (Dejonghe 1987, 1987MNRAS.224...13D, see also Breen et al. 2017, 2017MNRAS.471.2778B)
- plummer_isotropic_ic.py: Generates the particular case of an isotropic (in velocities) Plummer model (see Aarseth et al., 1974A&A....37..183A)
- read_snapshot.py: Class to read from a Swift .hdf5 snapshot file
- write_gadget.py: Wrapper used to write ics to a SWIFT-compatible hdf5 IC file (Josh Borrow (joshua.borrow@durham.ac.uk), see official SWIFT documentation)
