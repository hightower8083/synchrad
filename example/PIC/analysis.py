# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# %matplotlib notebook

import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from synchrad.calc import SynchRad

eph_keV_um = 1.24e-3

from opmd_viewer.addons.pic import LpaDiagnostics


def plot_colored(
    x, y, c, cmap=plt.cm.jet, vmin=None, vmax=None, steps=5, ax=None, **kw_args
):
    if ax is None:
        ax = plt.gca()

    v = np.asarray(c).copy()

    if vmax is None:
        vmax = v.max()

    if vmin is None:
        vmin = v.min()

    v -= vmin
    v /= vmax - vmin
    it = 0
    while it < c.size - steps:
        x_segm = x[it : it + steps + 1]
        y_segm = y[it : it + steps + 1]
        c_segm = cmap(v[it + steps // 2])
        ax.plot(x_segm, y_segm, c=c_segm, **kw_args)
        it += steps


# -

ts = LpaDiagnostics("./diags/hdf5/")
ts.get_field(
    "rho", plot=True, iteration=ts.iterations[-1], cmap="Blues_r", vmax=0, vmin=-5e6
)

# +
fl = h5.File("tracks.h5", mode="r")

Np = 800
dNt = 10
track_indcs = np.arange(fl["misc/N_particles"][()])

plt.figure(figsize=(9, 5), tight_layout=True)
for i in range(Np):
    coord = fl["tracks"][str(track_indcs[i])]["z"][()][::dNt]
    val = fl["tracks"][str(track_indcs[i])]["ux"][()][::dNt]
    col = fl["tracks"][str(track_indcs[i])]["uz"][()][::dNt]
    plot_colored(
        coord, val, col, steps=16, cmap=plt.cm.magma_r, lw=0.4, vmin=10, vmax=300
    )

plt.xlabel("Distance ($\mu$m)", fontsize=16)
plt.ylabel("$p_x$ ($m_e c$)", fontsize=16)
plt.savefig("trjectories.png")

# +
WeightTotal = 0.0
for ip in range(fl[f"misc/N_particles"][()]):
    WeightTotal += fl[f"tracks/{ip}/w"][()]

calc_input = {
    "grid": [(1e1, 1.2e4), (0, 0.03), (0.0, 2 * np.pi), (256, 36, 36)],
    "timeStep": fl["misc/cdt"][()],
    "dtype": "double",
    "native": False,
    "ctx": False,
}

calc = SynchRad(calc_input)
fl.close()

# +
E_cutoff = 1.0

file_spect = h5.File("spectrum.h5", mode="r")
calc.Data["radiation"] = file_spect["radiation"][()] / WeightTotal

spectral_axis = calc.get_spectral_axis()
energy_axis = spectral_axis * eph_keV_um
energy_axis_full = calc.Args["omega"] * eph_keV_um
spect_filter = (energy_axis_full > E_cutoff)[:, None, None]

energy_tot = calc.get_energy(lambda0_um=1.0, phot_num=False, spect_filter=spect_filter)
print(f"Total energy emitted in >{E_cutoff:g} keV: {energy_tot*1e15:g} pJ")

plt.figure()
energy_spectrum1D = calc.get_energy_spectrum(lambda0_um=1.0)
plt.semilogx(energy_axis, energy_spectrum1D)
plt.xlabel("Photon energy (keV)")
plt.ylabel("Brightness (ph./0.1%b.w./e$^-$)")
plt.savefig("energy_spectrum_1D.png")

# +
plt.figure()

spotXY_far, ext_far = calc.get_spot_cartesian(bins=(512, 512), lambda0_um=1.0)

ext_far *= 1e3
plt.imshow(spotXY_far.T, origin="lower", cmap=plt.cm.afmhot_r, extent=ext_far)

plt.xlabel("x-plane angle (mrad)")
plt.ylabel("y-plane angle (mrad)")
plt.savefig("real_space.png")
file_spect.close()
