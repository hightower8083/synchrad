import numpy as np
from synchrad.calc import SynchRad
from synchrad.utils import energy_1m_eV


if __name__ == "__main__":
    calc_input = {
        "grid": [ (10.0 / energy_1m_eV, 10e3 / energy_1m_eV), # from 10eV to 10keV
                  (0, 0.03),
                  (0.0, 2 * np.pi),
                  (256, 36, 36)],
        "Features":['logGrid',], # to plot log-scale x-axis later
        "dtype": "double",
        "native": False,
        # "ctx": [0, 0], set your context to avoid being asked
    }

    calc = SynchRad(calc_input)
    calc.calculate_spectrum(file_tracks="tracks.h5",
                            file_spectrum="spectrum.h5",)
