import numpy as np
from scipy.constants import m_e, c, e, epsilon_0, hbar
from scipy.constants import alpha as alpha_fs
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import h5py
from numba import njit

try:
    from tvtk.api import tvtk, write_data
    tvtk_installed = True
except (ImportError,):
    tvtk_installed = False

J_in_um = 2e6*np.pi*hbar*c

@njit
def record_particles_step(tracks, nsteps, it, it_start,
                          x, y, z, ux, uy, uz, id,
                          Np_select, dNp):
    for ip in range(Np_select):
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
    for ip in range(Np_select):

        if np.isnan(x[ip]):
            continue

        if it_start[ip]==0:
            it_start[ip] = it

        point = [ x[ip], y[ip], z[ip],
                  ux[ip], uy[ip], uz[ip], id[ip] ]

        tracks[nsteps[ip], ip, :] = point
        nsteps[ip] += 1

    return tracks, nsteps, it_start


class Utilities:

    def get_full_spectrum(self, spect_filter=None, \
                          phot_num=False, lambda0_um=None,
                          comp='total', iteration=-1):

        keys = self.Data['radiation'].keys()

        val = 0.0
        if comp=='total':
            for key in keys:
                val += self.Data['radiation'][key][iteration].astype(np.double)
        else:
            val += self.Data['radiation'][ comp][iteration].astype(np.double)

        if self.Args['mode'] == 'far':
            val = alpha_fs / (4 * np.pi**2) * val
        elif self.Args['mode'] == 'near':
            val = alpha_fs * np.pi / 4 * val
            val /= (2*np.pi)**2

        if spect_filter is not None:
            val *= spect_filter

        if phot_num:
            ax = self.Args['omega']
            val /= ax[:,None,None]
        else:
            val *= J_in_um
            if lambda0_um is None:
                print("Specify normalization wavelength in "+\
                  "microns (lambda0_um) ")
                return  np.zeros_like(val)
            val /= lambda0_um

        return val

    def get_energy_spectrum(self, spect_filter=None, \
      phot_num=False, lambda0_um=None, **kw_args):

        val = self.get_full_spectrum(spect_filter=spect_filter, \
          phot_num=phot_num, lambda0_um=lambda0_um, **kw_args)

        if self.Args['mode'] == 'far':
            val = 0.5*self.Args['dth']*self.Args['dph']*( (val[1:] + val[:-1]) \
              *np.sin(self.Args['theta'][None,:,None]) ).sum(-1).sum(-1)
        elif self.Args['mode'] == 'near2D':
            val = 0.5*self.Args['dx']*self.Args['dy'] \
                  *(val[1:] + val[:-1]).sum(-1).sum(-1)
        elif self.Args['mode'] == 'near':
            val = 0.5*self.Args['dr']*self.Args['dph']*( (val[1:] + val[:-1]) \
              *self.Args['radius'][None,:,None] ).sum(-1).sum(-1)

        return val

    def get_energy(self, spect_filter=None, \
      phot_num=False, lambda0_um=None, **kw_args):

        val = self.get_energy_spectrum(spect_filter=spect_filter, \
          phot_num=phot_num, lambda0_um=lambda0_um, **kw_args)

        val = (val*self.Args['dw']).sum()
        return val

    def get_spot(self, k0=None, spect_filter=None, \
      phot_num=False, lambda0_um = None, **kw_args):

        val = self.get_full_spectrum(spect_filter=spect_filter, \
          phot_num=phot_num, lambda0_um=lambda0_um,  **kw_args)

        if k0 is None:
            if val.shape[0]>1:
                val = 0.5*(val[1:] + val[:-1])
            val = (val*self.Args['dw'][:, None, None]).sum(0)
        else:
            ax = self.Args['omega']
            indx = (ax<k0).sum()
            if np.abs(self.Args['omega'][indx+1]-k0) \
              < np.abs(self.Args['omega'][indx]-k0):
                indx += 1
            val = val[indx]
        return val

    def get_spot_cartesian(self, k0=None, th_part=1.0, bins=(200, 200), \
      spect_filter=None, phot_num=False, lambda0_um = None, **kw_args):

        val = self.get_spot(spect_filter=spect_filter, \
          k0=k0, phot_num=phot_num, lambda0_um=lambda0_um, **kw_args)

        if self.Args['mode'] == 'far':
            th, ph = self.Args['theta'], self.Args['phi']
        elif self.Args['mode'] == 'near':
            th, ph = self.Args['radius'], self.Args['phi']
        else:
            print("This function is for 'far' and 'near' modes only")

        ph, th = np.meshgrid(ph,th)
        coord = ((th*np.cos(ph)).flatten(), (th*np.sin(ph)).flatten())

        th_max = th_part*th.max()
        new_coord = np.mgrid[ -th_max:th_max:bins[0]*1j, \
                              -th_max:th_max:bins[1]*1j ]

        val = griddata(coord, val.flatten(),
            (new_coord[0].flatten(), new_coord[1].flatten()),
            fill_value=0., method='linear'
          ).reshape(new_coord[0].shape)
        ext = np.array([-th_max,th_max,-th_max,th_max])
        return val, ext

    def get_spectral_axis(self):
        if 'wavelengthGrid' in self.Args['Features']:
            ax = 0.5*(self.Args['wavelengths'][1:] \
              + self.Args['wavelengths'][:-1])
        else:
            ax = 0.5*(self.Args['omega'][1:] + self.Args['omega'][:-1])
        return ax

    def exportToVTK( self, spect_filter=None, phot_num=False,\
                     lambda0_um = None, smooth_filter=None, \
                     filename='spectrum', project=False, **kw_args):

        if not tvtk_installed:
            print('TVTK API is not found')
            return

        omega, theta, phi = self.Args['omega'], self.Args['theta'], \
                            self.Args['phi']
        phi = np.r_[phi, 2*np.pi]

        if project is False:
            val = self.get_full_spectrum(spect_filter=spect_filter, \
                        phot_num=phot_num, lambda0_um=lambda0_um, **kw_args)
            scalar_name = 'spectrum'
        else:
            val = self.get_spot( phot_num=phot_num, lambda0_um=lambda0_um, \
                                 spect_filter=spect_filter, **kw_args)
            val = val[None, :, :]
            omega = omega[[-1]]
            filename += '_proj'
            scalar_name = 'spectrum_proj'

        val = np.concatenate( (val, val[:, :, [0]]), axis=-1 )\
            .astype(self.dtype)

        if smooth_filter is not None:
            val = gaussian_filter(val, smooth_filter)

        Nom = omega.size
        Nth = theta.size
        Nph = phi.size

        omega = omega[:,None,None].astype(self.dtype)
        theta = theta[None,:, None].astype(self.dtype)
        phi = phi[None,None,:].astype(self.dtype)

        x, y, z = (omega*np.sin(theta)*np.sin(phi)).ravel(), \
              (omega*np.sin(theta)*np.cos(phi)).ravel(), \
              (omega*np.cos(theta)*np.ones_like(phi)).ravel()

        spc_vtk = tvtk.StructuredGrid(dimensions=(Nph, Nth, Nom),
                    points=np.vstack((x.ravel(),y.ravel(),z.ravel())).T)

        spc_vtk.point_data.scalars = val.flatten()
        spc_vtk.point_data.scalars.name = scalar_name
        write_data(spc_vtk, filename)

def tracksFromOPMD(ts, pt, ref_iteration, fname=None, species=None,
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