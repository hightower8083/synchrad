import numpy as np
import h5py
from synchrad.calc import SynchRad

if __name__ == "__main__":
    calc_input = {
        "grid": [ (10 * 1e6, 1.2e4 * 1e6),
                  (0, 0.03),
                  (0.0, 2 * np.pi),
                  (256, 36, 36)],
        "dtype": "double",
        "native": False,
        "ctx": [0, 0],
    }

    calc = SynchRad(calc_input)
    calc.calculate_spectrum(file_tracks="tracks.h5",
                            file_spectrum="spectrum.h5",)