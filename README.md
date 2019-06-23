# Synchrotron Radiation calculator via openCL


## Overview

Tool to compute energy density of synchrotron radiation in the spectral volume using the Fourier transformed Lienard-Wiechert potentials (see eq. (14.65) of J.D.Jackson's _Classical electrodynamics_). 

This software processes the 3D (_x_, _y_, _z_, _px_, _py_, _pz_) trajectories of the charged point-like particles with _weights_,  and maps the emittied energy to the 3D spectral domain (_omega_, _theta_, _phi_). 
The package also contains a number of convenience methods for pre- and post-proccessing.

### Language and hardware

**SynchRad** is written in [openCL](https://www.khronos.org/opencl) and is interfaced with Python via [PyOpenCL](https://mathema.tician.de/software/pyopencl). 

Code also uses [Mako](https://github.com/sqlalchemy/mako) template manager for tweaking with the data types and _native_ functions. It was tested on GPU and CPU devices using NVIDIA, AMD and Apple platforms, and 
it demonstrates a reasonable performance on GPUs, while on CPU the simillar [OpenMP implementation](https://github.com/hightower8083/chimera) is signifincantly faster.

## Installation

Given that **openCL** and **PyOpenCL** are installed (e.g. via `conda` or `pip`) and configured on your machine, you can install **SynchRad** in a standard way, i.e. by cloning the source 
and running the `setup.py` script:
```
git clone https://github.com/hightower8083/synchrad.git
cd synchrad/
python setup.py install
```

To be able to use the software with multiple GPU or CPU devices via MPI one should also install `mpi4py`. To output result in VTK format via `exportToVTK` method, the `tvtk.api` should be installed.

## Usage

A minimal example of **SynchRad** usage can be found in `example/` folder of this repository.  

Another common example would be to calculate radiation produced by the particles from, for example, a PIC simulation.
In case if PIC software supports output in [OpenPMD standard](http://www.openpmd.org/#/start), this can be done with help of [openPMD-viewer](https://github.com/openPMD/openPMD-viewer), using the conversion function:
```python
from opmd_viewer import OpenPMDTimeSeries, ParticleTracker
from synchrad.utils import tracksFromOPMD

ts = LpaDiagnostics('./run/diags_track/hdf5/', check_all_files=False)
ref_iteration = ts.iterations[-1]
pt = ParticleTracker( ts, iteration=ref_iteration, 
                      preserve_particle_index=True)

tracksFromOPMD( ts, pt, ref_iteration=ref_iteration, fname='tracks.h5')
```
where one can see doc-lines of tracksFromOPMD for more optional arguments.

The prepared track can be processed and saved to file using following commands:
```python
import numpy as np
import h5py
from synchrad.calc import SynchRad

file_tracks = h5py.File('./tracks.h5', mode='r')

calc_input = {'grid':[ (1., 0.6e5),
                       (0,0.04),
                       (0.,2*np.pi),
                       (256, 32, 32) ],
              'timeStep':file_tracks['misc/cdt'][()],
              'dtype':'double',
              'ctx':'mpi',
             }

calc = SynchRad(calc_input)
calc.calculate_spectrum(h5_file=file_tracks)
file_tracks.close()

if calc.comm.rank==0:
    file_spect = h5py.File('./spectrum.h5', mode='w')
    file_spect['radiation'] = calc.Data['radiation']
    file_spect.close()
```
where radiation within 40 urad angle is calculated for the energies range [0, 74.4 keV].

For details on post-processing, one can see the example notebook in `example/`


## Author and Contributions

This software is developed by Igor A Andriyash (igor.andriyash@gmail.com), and it is on the early stage of development.

Everyone is welcome to contribute either by testing and benchmerking, or by introducing further optimizations and adding utility methods.
