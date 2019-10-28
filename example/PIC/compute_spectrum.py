import numpy as np
import h5py
from synchrad.calc import SynchRad

if __name__ == "__main__":
    file_tracks = h5py.File("tracks.h5", mode="r")

    calc_input = {
        "grid": [(1.0, 0.6e5), (0, 0.04), (0.0, 2 * np.pi), (256, 32, 32)],
        "timeStep": file_tracks["misc/cdt"][()],
        "dtype": "double",
        "native": True,
        "ctx": "mpi",
    }

    calc = SynchRad(calc_input)
    calc.calculate_spectrum(h5_file=file_tracks)

    file_tracks.close()

    if calc.comm.rank == 0:
        with h5py.File("spectrum.h5", "w") as f:
            f["radiation"] = calc.Data["radiation"]
