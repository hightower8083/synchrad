import numpy as np
import h5py
from synchrad.calc import SynchRad

if __name__ == "__main__":
    with h5py.File("tracks.h5", "r") as f:
        time_step = f["misc/cdt"][...]

    calc_input = {
        "grid": [(10, 1.2e4), (0, 0.03), (0.0, 2 * np.pi), (256, 36, 36)],
        "timeStep": time_step,
        "dtype": "double",
        "native": False,
        "ctx": None,
    }

    calc = SynchRad(calc_input)

    with h5py.File("tracks.h5", "r") as f:
        calc.calculate_spectrum(h5_file=f)

    if calc.rank == 0:
        with h5py.File("spectrum.h5", "w") as f:
            f["radiation"] = calc.Data["radiation"]
