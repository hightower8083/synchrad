import h5py
import numpy as np
from sliceplots import plot_multicolored_line, addcolorbar
from matplotlib import pyplot


if __name__ == "__main__":
    # load spectrum data
    f = h5py.File("tracks.h5", "r")

    number_of_particles = f["misc/N_particles"][...]
    track_index = range(number_of_particles)

    # plot particle trajectories
    fig, ax = pyplot.subplots(figsize=(9, 5))

    max = 0.0
    line = None

    for track_index in range(800):
        z = f[f"tracks/{track_index}/z"][...][::8]  # id, ux, uy, uz, w, x, y, z
        ux = f[f"tracks/{track_index}/ux"][...][::8]
        uz = f[f"tracks/{track_index}/uz"][...][::8]

        if np.max(uz) > max:
            max = np.max(uz)

        _, line = plot_multicolored_line(
            ax=ax, x=z, y=ux, other_y=uz, vmin=5, vmax=9, linewidth=0.4, alpha=0.3
        )
    ax.set(ylabel="$p_x$ [$m_e c$]", xlabel="$z$ [$\mu$m]", xlim=(30, 90), ylim=(-4, 4))
    cax = addcolorbar(ax=ax, mappable=line, label="$p_z$ [$m_e c$]")

    fig.savefig("colored_lines.png")
    print(max)

    f.close()

    # plot radiation spot in real space

    # plot far-field spectrum
