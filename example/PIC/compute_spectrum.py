import numpy as np
import h5py
from synchrad.calc import SynchRad

if __name__ == "__main__":
    with h5py.File("tracks.h5", "r") as f:
        time_step = f["misc/cdt"][()]

    calc_input = {
        "grid": [(1.0, 0.6e5), (0, 0.04), (0.0, 2 * np.pi), (256, 32, 32)],
        "timeStep": time_step,
        "dtype": "double",
        "native": True,
        # "ctx": "mpi",
    }
    calc = SynchRad(calc_input)

    with h5py.File("tracks.h5", "r") as f:
        calc.calculate_spectrum(h5_file=f)

    if calc.comm.rank == 0:
        with h5py.File("spectrum.h5", "w") as f:
            f["radiation"] = calc.Data["radiation"]
