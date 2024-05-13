import numpy as np
from scipy.constants import c
import h5py
import warnings

# try import numba and make dummy methods if cannot
try:
    from numba import njit, prange
    njit = njit(parallel=True)
except Exception:
    prange = range
    def njit(func):
        def func_wrp(*args, **kw_args):
            warnings.warn(f"Install Numba to get `{func.__name__}` " + \
                   "function greatly accelerated")
            return func(*args, **kw_args)
        return func_wrp

def tracksFromOPMD(ts, pt, ref_iteration,
                   fname='./tracks.h5',
                   Np_select=None, dNp=1,
                   sample_selection='random',
                   Nit_min=None, Nit_max=None,
                   z_is_xi=False,
                   shortest_track=8):

    species = pt.species
    all_pid = pt.selected_pid.copy()

    if Np_select is not None:
        if all_pid.size<Np_select:
            Np_select = all_pid.size
            print(f"Selected sample of {Np_select} tracks it too large. ",
              f"Only {all_pid.size} tracks are available in ParticleTracker")

        if sample_selection == 'random':
            selected_pid = np.random.choice(all_pid, size=Np_select)
        elif sample_selection == 'sequential':
            selected_pid = all_pid[:Np_select]
        else:
            print(f"Selected sampling method '{sample_selection}' is ",
                   "not available.")
    else:
        selected_pid = all_pid

    if dNp>1:
        selected_pid = selected_pid[::dNp]

    pt.__init__( ts, species=pt.species, iteration=ref_iteration,
                select=selected_pid, preserve_particle_index=True)

    iterations = ts.iterations.copy()
    t = ts.t.copy()
    iteration_ind = np.arange(iterations.size, dtype=np.int64)

    if Nit_min is not None:
        iteration_ind = iteration_ind[iterations>=Nit_min]

    if Nit_max is not None:
        iteration_ind = iteration_ind[iterations<=Nit_max]

    iterations = iterations[iteration_ind]
    t = t[iteration_ind]

    cdt = (t[1] - t[0]) * c
    cdt_array = (t[1:] - t[:-1]) * c

    TC = {}
    var_list = ['x', 'y', 'z', 'ux', 'uy', 'uz', 'w']

    TC['x'], TC['y'], TC['z'], TC['ux'], TC['uy'], TC['uz'], TC['w'] = \
        ts.iterate(ts.get_particle, select=pt, var_list=var_list, species=pt.species)

    # find and temporarily replace the non-consistent lists with NaNs
    for itLoc, valLoc in enumerate(TC['x']):
        if len(valLoc) != pt.N_selected:
            for var in var_list:
                TC[var][itLoc] = np.nan*np.empty(pt.N_selected)

    # convert lists to numpy arrays
    for var in var_list:
        TC[var] = np.array(TC[var], order='F').T

    # temporal patch to select iterations -- should go ts.iterate
    #for var in var_list:
    #    TC[var] = TC[var][:, iteration_ind]

    i_tr = 0
    it_start_global = np.inf
    it_end_global = 0
    f = h5py.File(fname, mode='w')

    for ip in range(pt.N_selected):
        track_pieces = split_track_by_nans(
                TC['x'][ip], TC['y'][ip], TC['z'][ip],
                TC['ux'][ip], TC['uy'][ip], TC['uz'][ip], TC['w'][ip])

        for track in track_pieces:
            x, y, z, ux, uy, uz, w, it_start = track
            nsteps = x.size
            if nsteps > shortest_track:
                f[f'tracks/{i_tr:d}/x'] = x
                f[f'tracks/{i_tr:d}/y'] = y
                if z_is_xi:
                    f[f'tracks/{i_tr:d}/z'] = z + c * t
                else:
                    f[f'tracks/{i_tr:d}/z'] = z
                f[f'tracks/{i_tr:d}/ux'] = ux
                f[f'tracks/{i_tr:d}/uy'] = uy
                f[f'tracks/{i_tr:d}/uz'] = uz
                f[f'tracks/{i_tr:d}/w'] = w
                f[f'tracks/{i_tr:d}/it_start'] = it_start

                it_end_local = it_start + nsteps
                if it_end_global < it_end_local:
                    it_end_global = it_end_local

                if it_start_global>it_start:
                    it_start_global = it_start

                i_tr += 1

    f['misc/cdt'] = cdt
    f['misc/cdt_array'] = cdt_array
    f['misc/N_particles'] = i_tr
    f['misc/it_range'] = np.array([it_start_global, it_end_global])
    f['misc/propagation_direction'] = 'z'
    f.close()

def tracksFromOPMD_old(ts, pt, ref_iteration, fname=None, species=None,
                   dNp=None, Np_select=None, maxRaduis=None,
                   Nit_min=None, Nit_max=None, verbose=True):

    Np = pt.N_selected
    w_select, = ts.get_particle(var_list=['w',], select=pt,
                                species=species, iteration=ref_iteration)

    if (Np_select is None) and (dNp is None):
        print('Either Np_select or dNp can be used. Choosing all particles.')
        Np_select = w_select.size

    if (Np_select is not None) and (dNp is not None):
        print("Only one of Np_select and dNp can be used. Stopping here.")
        return

    if Np_select is not None:
        w_select = w_select[:Np_select]

    if dNp is not None:
        w_select = w_select[::dNp]
        Np_select = w_select.size

    iterations = ts.iterations.copy()
    filter = np.ones_like(iterations)
    if Nit_min is not None:
        filter *= (iterations>Nit_min)
    if Nit_max is not None:
        filter *= (iterations<Nit_max)

    iter_ind_select = np.nonzero(filter)[0]
    iterations = iterations[iter_ind_select]
    Nt = iterations.size

    dt = (ts.t[1] - ts.t[0]) * c

    tracks = np.zeros( (Nt, Np_select, 7), dtype=np.double )
    nsteps = np.zeros( Np_select, dtype=np.int )
    it_start = np.zeros( Np_select, dtype=np.int )

    for it, iteration in enumerate(iterations):
        x, y, z, ux, uy, uz, id = ts.get_particle(
            var_list=['x', 'y', 'z', 'ux', 'uy', 'uz', 'id'],
            select=pt, iteration=iteration, species=species)

        if x.size < Np:
            continue

        if Np_select is not None:
            tracks, nsteps, it_start = record_particles_first(tracks, nsteps,
                it, it_start, x, y, z, ux, uy, uz, id, Np_select)
        elif dNp is not None:
            tracks, nsteps, it_start = record_particles_step(tracks, nsteps,
                it, it_start, x, y, z, ux, uy, uz, id, Np_select, dNp)

        if verbose:
            print( f"Done {it/len(iterations)*100: 0.1f}%",
                   end='\r', flush=True)

    if verbose: print( f"Done reading {Np_select} particles")
    if fname is not None:
        f = h5py.File(fname, mode='w')
        i_tr = 0
        it_end_glob = 0
        for ip in range(tracks.shape[1]):
            x, y, z, ux, uy, uz, id = tracks[:,ip,:].T
            if nsteps[ip]>8 :
                f[f'tracks/{i_tr:d}/x'] = x[:nsteps[ip]]
                f[f'tracks/{i_tr:d}/y'] = y[:nsteps[ip]]
                f[f'tracks/{i_tr:d}/z'] = z[:nsteps[ip]]
                f[f'tracks/{i_tr:d}/ux'] = ux[:nsteps[ip]]
                f[f'tracks/{i_tr:d}/uy'] = uy[:nsteps[ip]]
                f[f'tracks/{i_tr:d}/uz'] = uz[:nsteps[ip]]
                f[f'tracks/{i_tr:d}/id'] = id[:nsteps[ip]]
                f[f'tracks/{i_tr:d}/w'] = w_select[ip]
                f[f'tracks/{i_tr:d}/it_start'] = it_start[ip]

                if  it_start[ip] + nsteps[ip] > it_end_glob:
                    it_end_glob = it_start[ip] + nsteps[ip]

                i_tr += 1

        f['misc/cdt'] = dt
        f['misc/N_particles'] = i_tr
        f['misc/it_range'] = np.array([it_start.min(), it_end_glob])
        f['misc/propagation_direction'] = 'z'
        f.close()
        return
    else:
        particleTracks = []
        for ip in range(tracks.shape[1]):
            x, y, z, ux, uy, uz, id = tracks[:,ip,:].T
            if nsteps[ip]<8 :  continue
            particleTracks.append( [x[:nsteps[ip]], y[:nsteps[ip]],
                                    z[:nsteps[ip]], ux[:nsteps[ip]],
                                    uy[:nsteps[ip]], uz[:nsteps[ip]],
                                    w_select[ip], ], it_start[ip])

        return particleTracks, dt

def tracksFromVSIM(file_vsim, file_synchrad,
                   cdt, length_unit=1, dNit=1,
                   dNp=None, Np_select=None, verbose=True):

    dt = cdt/length_unit

    f_trk_orig = h5py.File(file_vsim, mode='r')
    f_trk_synch = h5py.File(file_synchrad, mode='w')

    Nt, Np, _ = f_trk_orig['tracks'].shape
    data_fields = f_trk_orig.keys()

    # unless there's any useful data
    w0 = 1.0

    N_tr = 0
    it_end_glob = 0
    it_start_glob = np.inf

    ip_indices = np.arange(Np)

    if dNp is not None:
        ip_indices = ip_indices[::dNp]

    if Np_select is not None:
        ip_indices = ip_indices[:Np_select]

    for ip in ip_indices:
        # switch axes for z-propagation
        z, y, x, uz, uy, ux = f_trk_orig['tracks'][::dNit, ip, :].T

        # discard out-of-box particles
        ind_select = np.nonzero(x>0)[0]
        it_start = ind_select[0]
        nsteps = ind_select.size

        x, y, z, ux, uy, uz = [v[ind_select] for v in [x, y, z, ux, uy, uz]]

        if it_start < it_start_glob:
            it_start_glob = it_start

        if len(z)>8 :
            f_trk_synch[f'tracks/{N_tr:d}/x'] = x/length_unit
            f_trk_synch[f'tracks/{N_tr:d}/y'] = y/length_unit
            f_trk_synch[f'tracks/{N_tr:d}/z'] = z/length_unit
            f_trk_synch[f'tracks/{N_tr:d}/ux'] = ux/c
            f_trk_synch[f'tracks/{N_tr:d}/uy'] = uy/c
            f_trk_synch[f'tracks/{N_tr:d}/uz'] = uz/c
            f_trk_synch[f'tracks/{N_tr:d}/w'] = w0
            f_trk_synch[f'tracks/{N_tr:d}/it_start'] = it_start
            N_tr += 1

            if  it_start + nsteps > it_end_glob:
                it_end_glob = it_start + nsteps

    f_trk_synch['misc/cdt'] = dt * dNit
    f_trk_synch['misc/N_particles'] = N_tr
    f_trk_synch['misc/it_range'] = np.array([it_start_glob, it_end_glob])
    f_trk_synch['misc/propagation_direction'] = 'z'

    f_trk_orig.close()
    f_trk_synch.close()
    if verbose:
        print(f'written {N_tr} tracks to {file_synchrad}')

def split_track_by_nans(x, y, z, ux, uy, uz, w):

    trackContainer = []

    x_loc = []
    y_loc = []
    z_loc = []
    ux_loc = []
    uy_loc = []
    uz_loc = []
    w_loc = []
    iterations_loc = []
    TrackRecorded = False

    for it, val in enumerate(w):
        if not np.isnan(val):
            TrackRecorded = False
            x_loc.append(x[it])
            y_loc.append(y[it])
            z_loc.append(z[it])
            ux_loc.append(ux[it])
            uy_loc.append(uy[it])
            uz_loc.append(uz[it])
            w_loc.append(w[it])
            iterations_loc.append(it)
        else:
            trackContainer.append([
                np.array(x_loc), np.array(y_loc), np.array(z_loc),
                np.array(ux_loc), np.array(uy_loc), np.array(uz_loc),
                np.array(w_loc), np.array(iterations_loc) ] )
            x_loc = []
            y_loc = []
            z_loc = []
            ux_loc = []
            uy_loc = []
            uz_loc = []
            w_loc = []
            iterations_loc = []
            TrackRecorded = True

    if TrackRecorded is False:
        trackContainer.append([
                np.array(x_loc), np.array(y_loc), np.array(z_loc),
                np.array(ux_loc), np.array(uy_loc), np.array(uz_loc),
                np.array(w_loc), np.array(iterations_loc) ] )
        TrackRecorded = True

    trackContainerSelected = []

    for i_track in range(len(trackContainer)):
        x_loc, y_loc, z_loc, ux_loc, uy_loc, uz_loc, w_loc, iterations_loc =\
            trackContainer[i_track]
        if x_loc.size>0:
            trackContainerSelected.append( [x_loc, y_loc, z_loc, \
                                      ux_loc, uy_loc, uz_loc, \
                                      w_loc[0], iterations_loc[0]])
    return trackContainerSelected

@njit
def record_particles_step(tracks, nsteps, it, it_start,
                          x, y, z, ux, uy, uz, id,
                          Np_select, dNp):
    for ip in prange(Np_select):
        ip_glob = ip*dNp

        if np.isnan(x[ip_glob]):
            continue

        if it_start[ip]==0:
            it_start[ip] = it

        point = [ x[ip_glob], y[ip_glob], z[ip_glob],
                  ux[ip_glob], uy[ip_glob], uz[ip_glob],
                  id[ip_glob] ]

        tracks[nsteps[ip], ip, :] = point
        nsteps[ip] += 1

    return tracks, nsteps, it_start

@njit
def record_particles_first(tracks, nsteps, it, it_start,
                           x, y, z, ux, uy, uz, id,
                           Np_select):
    for ip in prange(Np_select):

        if np.isnan(x[ip]):
            continue

        if it_start[ip]==0:
            it_start[ip] = it

        point = [ x[ip], y[ip], z[ip],
                  ux[ip], uy[ip], uz[ip], id[ip] ]

        tracks[nsteps[ip], ip, :] = point
        nsteps[ip] += 1

    return tracks, nsteps, it_start
