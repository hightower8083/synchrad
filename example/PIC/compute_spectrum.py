import numpy as np
import h5py
from synchrad.calc import SynchRad

if __name__ == "__main__":
    with h5py.File("tracks.h5", "r") as f:
        time_step = f["misc/cdt"][...]

        it_end_glob = f['misc/it_end'][()]
        it_start_glob = f['misc/it_start'][()]

    it_range = (it_start_glob, it_end_glob)

    calc_input = {
        "grid": [ (10, 1.2e4),
                  (0, 0.03),
                  (0.0, 2 * np.pi),
                  (256, 36, 36)],
        "timeStep": time_step,
        "dtype": "double",
        "native": False,
        "ctx": 'mpi',
    }

    calc = SynchRad(calc_input)

    with h5py.File("tracks.h5", "r") as f:
        calc.calculate_spectrum(h5_file=f, it_range=it_range)

    if calc.rank == 0:
        with h5py.File("spectrum.h5", "w") as f:

            for key in calc.Data['radiation'].keys():
                f['radiation/'+key] = calc.Data['radiation'][key]

            f['it_start'] = it_start_glob
            f['it_end'] = it_end_glob
            f['snap_iterations'] = calc.snap_iterations.get()

