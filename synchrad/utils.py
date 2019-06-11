import numpy as np
from scipy.constants import m_e, c, e, epsilon_0, hbar
from scipy.constants import alpha as alpha_fs
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

try:
    from tvtk.api import tvtk, write_data
    tvtk_installed = True
except (ImportError,):
    tvtk_installed = False

J_in_um = 2e6*np.pi*hbar*c


class Utilities:

    def get_full_spectrum(self, spect_filter=None, \
      phot_num=False, lambda0_um=None):

        if self.Args['mode'] == 'far':
            val = alpha_fs/(4*np.pi**2)*self.Data['radiation'].astype(np.double)
        elif self.Args['mode'] == 'near2D' or self.Args['mode'] == 'near':
            val = alpha_fs*np.pi/4*self.Data['radiation'].astype(np.double)
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
      phot_num=False, lambda0_um=None):

        val = self.get_full_spectrum(spect_filter=spect_filter, \
          phot_num=phot_num, lambda0_um=lambda0_um)

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
      phot_num=False, lambda0_um=None):

        val = self.get_energy_spectrum(spect_filter=spect_filter, \
          phot_num=phot_num, lambda0_um=lambda0_um)

        val = (val*self.Args['dw']).sum()
        return val

    def get_spot(self, k0=None, spect_filter=None, \
      phot_num=False, lambda0_um = None):

        val = self.get_full_spectrum(spect_filter=spect_filter, \
          phot_num=phot_num, lambda0_um=lambda0_um)

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
      spect_filter=None, phot_num=False, lambda0_um = None):

        val = self.get_spot(spect_filter=spect_filter, \
          k0=k0, phot_num=phot_num, lambda0_um=lambda0_um)

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
                     filename='spectrum', project=False):

        if not tvtk_installed:
            print('TVTK API is not found')
            return

        omega, theta, phi = self.Args['omega'], self.Args['theta'], \
                            self.Args['phi']
        phi = np.r_[phi, 2*np.pi]

        if project is False:
            val = self.get_full_spectrum(spect_filter=spect_filter, \
                        phot_num=phot_num, lambda0_um=lambda0_um)
            scalar_name = 'spectrum'
        else:
            val = self.get_spot( phot_num=phot_num, lambda0_um=lambda0_um, \
                                 spect_filter=spect_filter)
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

def tracksFromOPMD(ts, pt, ref_iteration, dNp=1, verbose=True):

    w_select, = ts.get_particle(var_list=['w',], select=pt,
                                iteration=ref_iteration )
    w_select = w_select[::dNp]
    Np = pt.N_selected
    Nt = ts.iterations.size
    Np_select = Np//dNp

    dt = (ts.t[1] - ts.t[0]) * c * 1e6 # in microns as coordinates

    tracks = np.zeros( (Np_select, 6, Nt), dtype=np.double )
    nsteps = np.zeros( Np_select, dtype=np.int )

    for iteration in ts.iterations:
        x, y, z, ux, uy, uz, w = ts.get_particle(
            var_list=['x', 'y', 'z', 'ux', 'uy', 'uz', 'w'],
            select=pt, iteration=iteration )

        if w.shape[0] < Np: continue
        for ip_select in range(Np_select):
            ip_glob = ip_select*dNp
            if np.isnan(x[ip_glob]): continue
            point = [ x[ip_glob], y[ip_glob], z[ip_glob],
                      ux[ip_glob], uy[ip_glob], uz[ip_glob] ]

            tracks[ip_select, :, nsteps[ip_select]] = point
            nsteps[ip_select] += 1

        if verbose:
            print( "Done {:0.1f}%".format(iteration/ts.iterations[-1] * 100),
                   end='\r', flush=True)

    particleTracks = []
    for ip, track in enumerate(tracks):
        x, y, z, ux, uy, uz = track
        particleTracks.append( [x[:nsteps[ip]], y[:nsteps[ip]],
                                z[:nsteps[ip]], ux[:nsteps[ip]],
                                uy[:nsteps[ip]], uz[:nsteps[ip]],
                                w_select[ip]] )

    return particleTracks, dt
