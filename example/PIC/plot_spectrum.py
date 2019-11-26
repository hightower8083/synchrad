import h5py
import numpy as np
from matplotlib import pyplot
from opmd_viewer.addons.pic import LpaDiagnostics
from sliceplots import plot_multicolored_line, addcolorbar, Plot2D

from synchrad.calc import SynchRad

if __name__ == "__main__":
    # plot electron density
    ts = LpaDiagnostics("./diags/hdf5/")
    ts.get_field(
        "rho", plot=True, iteration=ts.iterations[-1], vmax=0, vmin=-5e6, cmap="Blues_r"
    )
    pyplot.gcf().savefig("rho.png")

    # plot particle trajectories
    fig, ax = pyplot.subplots(figsize=(9, 5))

    with h5py.File("tracks.h5", "r") as f:
        for track_index in range(800):
            z = f[f"tracks/{track_index}/z"][...][::10]
            ux = f[f"tracks/{track_index}/ux"][...][::10]
            uz = f[f"tracks/{track_index}/uz"][...][::10]

            _, line = plot_multicolored_line(
                ax=ax,
                x=z,
                y=ux,
                other_y=uz,
                vmin=10,
                vmax=300,
                linewidth=0.4,
                alpha=0.3,
                cmap=pyplot.cm.magma_r,
            )
    ax.set(
        ylabel="$p_x$ [$m_e c$]", xlabel="$z$ [$\mu$m]", xlim=(50, 350), ylim=(-15, 15)
    )
    cax = addcolorbar(ax=ax, mappable=line, label="$p_z$ [$m_e c$]")

    fig.savefig("particle_trajectories.png")

    # load spectrum data
    with h5py.File("tracks.h5", "r") as f:
        number_of_particles = f["misc/N_particles"][...]
        time_step = f["misc/cdt"][...]

        total_particle_weight = 0.0
        for particle_index in range(number_of_particles):
            total_particle_weight += f[f"tracks/{particle_index}/w"][...]

    calc_input = {
        "grid": [(1e1, 1.2e4), (0, 0.03), (0.0, 2 * np.pi), (256, 36, 36)],
        "timeStep": time_step,
        "dtype": "double",
        "native": False,
        "ctx": False,
    }
    calc = SynchRad(calc_input)

    with h5py.File("spectrum.h5", "r") as f:
        calc.Data["radiation"] = f["radiation"][...] / total_particle_weight

    # compute total emitted energy
    E_cutoff = 1.0  # keV
    eph_keV_um = 1.24e-3

    spectral_axis = calc.get_spectral_axis()
    energy_axis = spectral_axis * eph_keV_um
    energy_axis_full = calc.Args["omega"] * eph_keV_um
    spect_filter = (energy_axis_full > E_cutoff)[:, np.newaxis, np.newaxis]

    energy_tot = calc.get_energy(
        lambda0_um=1.0, phot_num=False, spect_filter=spect_filter
    )
    print(f"Total energy emitted in >{E_cutoff:g} keV: {energy_tot * 1e15:g} pJ")

    # plot energy spectrum
    pyplot.figure()
    energy_spectrum1D = calc.get_energy_spectrum(lambda0_um=1.0)
    pyplot.semilogx(energy_axis, energy_spectrum1D)
    pyplot.xlabel("Photon energy (keV)")
    pyplot.ylabel("Brightness (ph./0.1%b.w./e$^-$)")
    pyplot.savefig("energy_spectrum_1D.png")

    # plot real-space emission
    spotXY_far, ext_far = calc.get_spot_cartesian(bins=(512, 512), lambda0_um=1.0)

    fig = pyplot.figure(figsize=(8, 8))
    Plot2D(
        fig=fig,
        arr2d=spotXY_far.T,
        h_axis=np.linspace(ext_far[0], ext_far[1], 512) * 1e3,
        v_axis=np.linspace(ext_far[2], ext_far[3], 512) * 1e3,
        xlabel="x-plane angle (mrad)",
        ylabel="y-plane angle (mrad)",
        hslice_val=0.0,
        vslice_val=0.0,
        hslice_opts={"color": "#1f77b4", "lw": 1.5, "ls": "-"},
        vslice_opts={"color": "#d62728", "ls": "-"},
        cmap=pyplot.cm.afmhot_r,
    )
    fig.savefig("real_space.png")
