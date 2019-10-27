import numpy as np
from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable

from example.undulator.api import undulator_spectrum


def analytic_check(spectrum):
    analytical_spectrum = spectrum.energy_analytical()
    energy_model = spectrum.synch_rad.get_energy(lambda0_um=1.0)
    var = abs(energy_model - analytical_spectrum) / analytical_spectrum
    print(f"Deviation from analytic estimate is {100 * var:.2f}")


if __name__ == "__main__":
    spectrum = undulator_spectrum(number_of_particles=10)
    analytic_check(spectrum)

    # Plot the spot observed with a band-filter
    kFilter = 0.93 * spectrum.central_wavenumber
    kBand = 0.003 * kFilter
    k = spectrum.synch_rad.Args["omega"][:, None, None]
    spect_filter = np.exp(-(k - kFilter) ** 2 / kBand ** 2)

    spot, extent = spectrum.synch_rad.get_spot_cartesian(
        bins=(600, 600), lambda0_um=2e4, th_part=0.2, spect_filter=spect_filter
    )

    fig, ax = pyplot.subplots()
    ax.imshow(spot.T, extent=extent * 1e3, cmap=pyplot.cm.nipy_spectral)
    ax.set_xlabel("mrad", size=14)
    ax.set_ylabel("mrad", size=14)
    fig.savefig("spot.png")

    fig, ax = pyplot.subplots()
    img = ax.imshow(
        spectrum.synch_rad.Data["radiation"].mean(-1).T, origin="lower", aspect="auto"
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)
    fig.savefig("radiation.png")

    # Results on GeForce GTX 950M, OpenCL 1.2, farfield from 24 particles
    # single precision, native functions: 35s, 1.83% error
