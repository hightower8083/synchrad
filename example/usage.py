import numpy as np
from matplotlib import pyplot

from undulator.api import undulator_spectrum
import xarray as xr

import logging

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S", level=logging.INFO
)


def analytic_check(spectrum):
    analytical_spectrum = spectrum.energy_analytical()
    energy_model = spectrum.synch_rad.get_energy(lambda0_um=1.0)
    var = abs(energy_model - analytical_spectrum) / analytical_spectrum
    logging.info(f"Deviation from analytic estimate is {100 * var:.2f}")


if __name__ == "__main__":
    spectrum = undulator_spectrum(number_of_particles=10)
    analytic_check(spectrum)

    # Plot the spot observed with a band-filter
    k_filter = 0.93 * spectrum.central_wavenumber
    k_band = 0.003 * k_filter
    k = spectrum.grid.k[:, np.newaxis, np.newaxis]
    filter = np.exp(-(k - k_filter) ** 2 / k_band ** 2)

    # the SynchRad class should return xarray objects, complete with axes and attached units, as shown below
    # then the user can do data processing on these objects, and plot them via different backends,
    # convert to numpy arrays etc. xarray also supports dask for chunk processing of large arrays
    # TODO: refactor, no clue what this line is doing
    spot, extent = spectrum.synch_rad.get_spot_cartesian(
        bins=(600, 600), lambda0_um=2e4, th_part=0.2, spect_filter=filter
    )
    spotxr = xr.DataArray(
        spot,
        dims=["x", "y"],
        coords=dict(
            x=np.linspace(extent[0], extent[1], 600) * 1e3,
            y=np.linspace(extent[2], extent[3], 600) * 1e3,
        ),
        attrs=dict(long_name="radiation", units="units"),
    ).transpose()
    spotxr.x.attrs["units"] = "mrad"
    spotxr.y.attrs["units"] = "mrad"
    #
    pyplot.figure()
    spotxr.plot(cmap=pyplot.cm.nipy_spectral)

    radiation = xr.DataArray(
        spectrum.synch_rad.Data["radiation"],
        dims=["k", "theta", "phi"],
        coords=dict(
            k=spectrum.grid.k, theta=spectrum.grid.theta, phi=spectrum.grid.phi
        ),
        attrs=dict(long_name="radiation", units="units"),
    )
    radiation.k.attrs["units"] = "k units"
    radiation.theta.attrs["units"] = "rad"
    radiation.phi.attrs["units"] = "rad"

    radiation_k_theta = radiation.mean(dim="phi", keep_attrs=True).transpose()
    pyplot.figure()
    radiation_k_theta.plot()
