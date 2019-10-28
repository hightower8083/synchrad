import logging

import numpy as np
import xarray as xr
from matplotlib import pyplot

from undulator.api import undulator_spectrum, Resolution

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S", level=logging.INFO
)


def analytic_check(some_spectrum):
    analytical_spectrum = some_spectrum.energy_analytical()
    energy_model = some_spectrum.synch_rad.get_energy(lambda0_um=1.0)
    var = abs(energy_model - analytical_spectrum) / analytical_spectrum
    logging.info(f"Deviation from analytic estimate is {100 * var:.2f}")


if __name__ == "__main__":
    # compute undulator spectrum
    spectrum = undulator_spectrum(
        number_of_particles=10, opencl_context=(0,), resolution=Resolution(phi=32)
    )

    # compare with analytic result
    analytic_check(spectrum)

    # plot particle trajectories
    _, ax = pyplot.subplots()
    for particle in spectrum:
        ax.plot(particle.track.z, particle.track.x * 1e6, alpha=0.7)
    ax.set(xlabel="z", ylabel=r"x [$\mu$m]", title="Particle trajectories")

    # plot radiation spot in real space
    k_filter = 0.93 * spectrum.central_wavenumber
    k_band = 0.003 * k_filter
    k = spectrum.grid.k[:, np.newaxis, np.newaxis]
    band_pass = np.exp(-(k - k_filter) ** 2 / k_band ** 2)

    # if the SynchRad class would return xarray objects, complete with axes and attached units, as shown below
    # then the user could do data processing on these objects, and plot them via different backends,
    # convert to numpy arrays etc. xarray also supports dask for chunk processing of large arrays.
    spot, extent = spectrum.synch_rad.get_spot_cartesian(
        bins=(600, 600), lambda0_um=2e4, th_part=0.2, spect_filter=band_pass
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
    pyplot.gca().set(title="Band-filtered spot")

    # plot far-field spectrum
    radiation = xr.DataArray(
        spectrum.synch_rad.Data["radiation"],
        dims=["k", "theta", "phi"],
        coords=dict(
            k=spectrum.grid.k, theta=spectrum.grid.theta, phi=spectrum.grid.phi
        ),
        attrs=dict(long_name="radiation", units="units"),
    )
    radiation.k.attrs["units"] = "units"
    radiation.theta.attrs["units"] = "rad"
    radiation.phi.attrs["units"] = "rad"

    radiation_k_theta = radiation.mean(dim="phi", keep_attrs=True).transpose()
    pyplot.figure()
    radiation_k_theta.plot()
    pyplot.gca().set(title="Far-field radiation spectrum")
