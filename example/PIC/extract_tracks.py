from opmd_viewer import ParticleTracker
from opmd_viewer.addons import LpaDiagnostics

from synchrad.utils import tracksFromOPMD


if __name__ == "__main__":
    ts = LpaDiagnostics("diags_track/hdf5", check_all_files=True)
    ref_iteration = ts.iterations[-1]

    pt = ParticleTracker(ts, iteration=ref_iteration, preserve_particle_index=True)
    tracksFromOPMD(ts, pt, ref_iteration=ref_iteration, fname="tracks.h5")
