import numpy as np
import h5py
import pyopencl as cl
import pyopencl.array as arrcl
from mako.template import Template

from .utils import Utilities
from synchrad import __path__ as src_path

try:
    from mpi4py import MPI
    mpi_installed = True
except (ImportError):
    mpi_installed = False

src_path = src_path[0] + '/'


class SynchRad(Utilities):

    def __init__(self, Args={}, file_spectrum=None):

        if mpi_installed:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.rank
            self.size = self.comm.size
        else:
            self.rank = 0
            self.size = 1
        if file_spectrum is None:
            self._init_args(Args)
            self._init_comm()
            self._init_data()
            self._compile_kernels()
        else:
            self._read_args(file_spectrum)

    def calculate_spectrum(self, particleTracks=[], file_tracks=None,
                           timeStep=None, comp='total', Np_max=None,
                           nSnaps=1, it_range=None, file_spectrum=None,
                           verbose=True):

        nSnaps = np.uint32(nSnaps)
        self._init_raditaion(comp, nSnaps)

        if timeStep is not None:
            self.Args['timeStep'] = self.dtype(timeStep)

        if it_range is not None:
            it_range = tuple(it_range)

        # input from a file
        if file_tracks is not None:
            f_tracks = h5py.File(file_tracks, "r")

            self.Args['timeStep'] = self.dtype(f_tracks["misc/cdt"][()])

            # determine if it_range is provided
            if it_range is None:
                if 'it_range' in f_tracks['misc'].keys():
                    it_range = tuple(f_tracks['misc/it_range'][()])
                    self._set_snap_iterations(it_range, nSnaps)
                    if self.rank==0 and verbose:
                        print("it_range from the input file will be used")
                else:
                    if self.rank==0 and verbose:
                        print("Separate it_range for each track will be used")
            else:
                self._set_snap_iterations(it_range, nSnaps)

            # set number of tracks to process
            Np = f_tracks['misc/N_particles'][()]
            if Np_max is not None:
                Np = min(Np_max, Np)

            # load all tracks for each MPI node
            particleTracks = []
            cmps = ('x', 'y', 'z', 'ux', 'uy', 'uz', 'w', 'it_start')
            part_ind = np.arange(Np)[self.rank::self.size]
            for ip in part_ind:
                track = [f_tracks[f"tracks/{ip:d}/{cmp}"][()] for cmp in cmps]
                particleTracks.append(track)

            if self.rank==0 and verbose:
                print("Tracks are loaded")
            f_tracks.close()

        # input from a list
        else:
            # determine if it_range is provided
            if it_range is not None:
                self._set_snap_iterations(it_range, nSnaps)
            else:
                if self.rank==0 and verbose:
                    print("Separate it_range for each track will be used")

            # set number of tracks to process
            Np = len(particleTracks)
            if Np_max is not None:
                Np = min(Np_max, Np)

            # take set of tracks for each MPI node
            particleTracks = particleTracks[:Np][self.rank::self.size]

        # process the tracks
        itr = 0
        self.total_weight = 0.0
        for track in particleTracks:
            self.total_weight += track[6]
            track = self._track_to_device(track)
            self._process_track(track, comp, nSnaps, it_range)
            itr += 1
            if self.rank==0 and verbose:
                progress = itr/len(particleTracks) * 100
                print("Done {:0.1f}%".format(progress),
                      end='\r', flush=True)

        # receive and gather tracks from devices
        self._spectr_from_device(nSnaps)
        if mpi_installed:
            self._gather_result_mpi()

        if file_spectrum is not None:
            if self.rank == 0:
                f_out = h5py.File(file_spectrum, "w")
                for key in self.Data['radiation'].keys():
                    f_out['radiation/'+key] = self.Data['radiation'][key]

                ArgsKeys = list(self.Args.keys())
                ArgsKeys.remove('grid')
                ArgsKeys.remove('ctx')
                ArgsKeys.remove('Features')
                for key in ArgsKeys:
                    f_out['Args/'+key] = self.Args[key]

                if len(self.Args['Features'])>0:
                    for key in self.Args['Features'].keys():
                        f_out['Args/Features'+key] = \
                          self.Args['Features'][key]

                f_out['snap_iterations'] = self.snap_iterations.get()
                f_out['total_weight'] = self.total_weight

                print(f"Spectrum is saved to {file_spectrum}")
                f_out.close()

    def _process_track(self, particleTrack, comp, nSnaps, it_range):

        x, y, z, ux, uy, uz, wp, it_start = particleTrack

        # set individual it_range, if not defined
        if it_range is None:
            # it_start is set to 0
            it_start = np.uint32(0)
            it_range = (0, x.size)
            self._set_snap_iterations(it_range, nSnaps)

        spect = self.Data['radiation']
        WGS, WGS_tot = self._get_wgs(self.Args['numGridNodes'])

        args_track = [coord.data for coord in (x, y, z, ux, uy, uz)]
        args_track += [wp, it_start, np.uint32(it_range[-1]), np.uint32(x.size)]

        if self.Args['mode'] is 'far':
            axs_str = ('omega', 'sinTheta', 'cosTheta', 'sinPhi', 'cosPhi')
        elif self.Args['mode'] is 'near':
            axs_str = ('omega', 'radius', 'sinPhi', 'cosPhi')

        args_axes = [self.Data[name].data for name in axs_str]

        if self.Args['mode'] is 'near':
            args_axes += [ self.dtype(self.Args['grid'][3]), ]

        args_res = [np.uint32(Nn) for Nn in self.Args['gridNodeNums']]
        args_aux = [self.Args['timeStep'], nSnaps, self.snap_iterations.data]

        args = args_track + args_axes + args_res + args_aux

        if comp is 'total':
            self._mapper.total(self.queue, (WGS_tot, ), (WGS, ),
                               spect['total'].data, *args)

        elif comp is 'cartesian':
            self._mapper.cartesian_comps(self.queue, (WGS_tot, ), (WGS, ),
                spect['x'].data, spect['y'].data, spect['z'].data, *args)

        elif comp is 'spheric':
            self._mapper.spheric_comps(self.queue, (WGS_tot, ), (WGS, ),
                spect['r'].data, spect['theta'].data, spect['phi'].data, *args)

    def _init_args(self, Args):
        self.Args = Args

        if 'mode' not in self.Args.keys():
            self.Args['mode'] = 'far'

        if 'dtype' not in self.Args:
            self.Args['dtype'] = 'double'

        if self.Args['dtype'] is 'double':
            self.dtype = np.double
        elif self.Args['dtype'] is 'float':
            self.dtype = np.single

        if self.Args['dtype'] is 'float':
            if self.Args['mode'] is 'far':
                print ( 'WARNING: Chosen single precision should be ' + \
                        'used with care for the farfield calculations\n' )
            elif self.Args['mode'] is 'near':
                print ( 'WARNING: Chosen single precision is not ' + \
                        'recommended for the nearfield calculations\n' )

        if 'ctx' not in self.Args.keys():
            self.Args['ctx'] = None

        if 'Features' not in self.Args.keys():
            self.Args['Features'] = {}

        self.Args['gridNodeNums'] = self.Args['grid'][-1]

        self.Args['numGridNodes'] = int(np.prod(self.Args['gridNodeNums']))

        omega_min, omega_max = self.Args['grid'][0]
        No = self.Args['gridNodeNums'][0]
        if 'wavelengthGrid' in self.Args['Features']:
            self.Args['wavelengths'] = np.r_[1./omega_max:1./omega_min:No*1j]
            omega = 1./self.Args['wavelengths']
        elif 'logGrid' in self.Args['Features']:
            d_log_w = np.log(omega_max/omega_min) / (No-1.0)
            omega = omega_min * np.exp( d_log_w*np.arange(No)  )
        else:
            omega = np.r_[omega_min:omega_max:No*1j]

        self.Args['omega'] = omega.astype(self.dtype)

        if No>1:
            self.Args['dw'] = np.abs( omega[1:]-omega[:-1] )
        else:
            self.Args['dw'] = np.array([1.,], dtype=self.dtype)

        if self.Args['mode'] == 'far':
            Nt, Np = self.Args['gridNodeNums'][1:]
            theta_min, theta_max  = self.Args['grid'][1]
            phi_min, phi_max = self.Args['grid'][2]

            theta = np.r_[theta_min:theta_max:Nt*1j]
            phi = phi_min + (phi_max-phi_min)/Np*np.arange(Np)

            if Nt>1:
                self.Args['dth'] = theta[1] - theta[0]
            else:
                self.Args['dth'] = self.dtype(1.)

            if Np>1:
                self.Args['dph'] = phi[1] - phi[0]
            else:
                self.Args['dph'] = self.dtype(1.)

            self.Args['theta'] = theta.astype(self.dtype)
            self.Args['phi'] = phi.astype(self.dtype)
            self.Args['dV'] = self.Args['dw']*self.Args['dth']*self.Args['dph']

        elif self.Args['mode'] == 'near':
            Nr, Np = gridNodeNums[1:]

            r_min, r_max  = self.Args['grid'][1]
            phi_min, phi_max = self.Args['grid'][2]

            radius = np.r_[r_min:r_max:Nr*1j]
            phi = phi_min + (phi_max-phi_min)/Np*np.arange(Np)

            if Nr>1:
                self.Args['dr'] = radius[1] - radius[0]
            else:
                self.Args['dr'] = 1.

            if Np>1:
                self.Args['dph'] = phi[1] - phi[0]
            else:
                self.Args['dph'] = 1.

            self.Args['phi'] = phi.astype(self.dtype)
            self.Args['radius'] = radius.astype(self.dtype)
            self.Args['dV'] = self.Args['dw']*self.Args['dr']*self.Args['dph']

    def _init_raditaion(self, comp, nSnaps):

        radiation_shape = tuple(self.Args['gridNodeNums'][::-1])
        radiation_shape = (nSnaps, ) + radiation_shape

        vec_comps = {'cartesian':['x', 'y', 'z'],
                     'spheric':['r', 'theta', 'phi'],
                     'total': ['total',]
                     }

        self.Data['radiation'] = {}

        for vec_comp in vec_comps[comp]:
            self.Data['radiation'][vec_comp] = \
                arrcl.zeros(self.queue, radiation_shape, dtype=self.dtype)

    def _init_data(self):

        self.Data = {}

        if self.plat_name is "None":
            return

        # Note that dw is not multiplied by 2*pi
        self.Data['omega'] = arrcl.to_device( self.queue,
                                 self.dtype(2*np.pi)*self.Args['omega'] )

        if self.Args['mode'] is 'far':
            self.Data['sinTheta'] = arrcl.to_device( self.queue,
                                                 np.sin(self.Args['theta']) )
            self.Data['cosTheta'] = arrcl.to_device( self.queue,
                                                 np.cos(self.Args['theta']) )
            self.Data['sinPhi'] = arrcl.to_device( self.queue,
                                                   np.sin(self.Args['phi']) )
            self.Data['cosPhi'] = arrcl.to_device( self.queue,
                                                   np.cos(self.Args['phi']) )
        elif self.Args['mode'] is 'near':
            self.Data['radius'] = arrcl.to_device( self.queue,
                                                   self.Args['radius'] )
            self.Data['sinPhi'] = arrcl.to_device( self.queue,
                                                   np.sin(self.Args['phi']) )
            self.Data['cosPhi'] = arrcl.to_device( self.queue,
                                                   np.cos(self.Args['phi']) )
    def _init_comm(self):

        ctx_kw_args = {}
        if self.Args['ctx'] is None:
            ctx_kw_args['interactive'] = True
        elif self.Args['ctx'] is 'mpi':
            # temporal definition, assumes default 0th platform
            ctx_kw_args['answers'] = [0, self.rank]
        elif self.Args['ctx'] is not False:
            ctx_kw_args['answers'] = self.Args['ctx']

        if self.Args['ctx'] is False:
            self.dev_type = "Starting without"
            self.dev_name = ""
            self.plat_name = "None"
            self.ocl_version = "None"
        else:
            try:
                self.ctx = cl.create_some_context(**ctx_kw_args)
                self.queue = cl.CommandQueue(self.ctx)

                selected_dev = self.queue.device
                self.dev_type = cl.device_type.to_string(selected_dev.type)
                self.dev_name = self.queue.device.name

                self.plat_name = selected_dev.platform.vendor
                self.ocl_version = selected_dev.opencl_c_version
            except:
                self.dev_type = "Starting without"
                self.dev_name = ""
                self.plat_name = "None"
                self.ocl_version = "None"

        msg = "  {} device: {}".format(self.dev_type, self.dev_name)
        if self.size>1:
            msg = self.comm.gather(msg)
        else:
            msg = [msg,]

        if self.rank==0:
            print("Running on {} devices".format(self.size))
            for s in msg: print(s)
            print("Platform: {}\nCompiler: {}".\
                   format(self.plat_name, self.ocl_version) )

        self._set_global_working_group_size()

    def _gather_result_mpi(self):

        for key in self.Data['radiation'].keys():
            buff = np.zeros_like(self.Data['radiation'][key])
            self.comm.barrier()
            self.comm.Reduce([self.Data['radiation'][key], MPI.DOUBLE],
                             [buff, MPI.DOUBLE])
            self.comm.barrier()
            self.Data['radiation'][key] = buff

        self.comm.barrier()
        self.total_weight = self.comm.reduce(self.total_weight)

    def _spectr_from_device(self, nSnaps):
        for key in self.Data['radiation'].keys():
            buff = self.Data['radiation'][key].get().swapaxes(-1,-3)
            #if nSnaps == 1:
            #    buff = buff[-1]
            self.Data['radiation'][key] = np.ascontiguousarray(buff, dtype=np.double)

    def _track_to_device(self, particleTrack):
        if len(particleTrack) == 8:
            x, y, z, ux, uy, uz, wp, it_start = particleTrack
        elif len(particleTrack) == 7:
            x, y, z, ux, uy, uz, wp = particleTrack
            it_start = 0

        x = arrcl.to_device( self.queue,
            np.ascontiguousarray(x.astype(self.dtype)) )
        y = arrcl.to_device( self.queue,
            np.ascontiguousarray(y.astype(self.dtype)) )
        z = arrcl.to_device( self.queue,
            np.ascontiguousarray(z.astype(self.dtype)) )
        ux = arrcl.to_device( self.queue,
            np.ascontiguousarray(ux.astype(self.dtype)) )
        uy = arrcl.to_device( self.queue,
            np.ascontiguousarray(uy.astype(self.dtype)) )
        uz = arrcl.to_device( self.queue,
            np.ascontiguousarray(uz.astype(self.dtype)) )
        wp = self.dtype(wp)
        it_start = np.uint32(it_start)

        particleTrack = [x, y, z, ux, uy, uz, wp, it_start]

        return particleTrack

    def _compile_kernels(self):

        if self.plat_name is "None":
            return

        agrs = {}
        agrs['my_dtype'] = self.Args['dtype']
        if 'native' in self.Args:
            agrs['f_native'] = 'native_'
        else:
            agrs['f_native'] = ''

        fname = src_path
        if self.Args['mode'] is 'far':
            fname += "kernel_farfield.cl"
        elif self.Args['mode'] is 'near':
            fname += "kernel_nearfield.cl"

        src = Template( filename=fname ).render(**agrs)
        self._mapper = cl.Program(self.ctx, src).build()

    def _set_snap_iterations(self, it_range, nSnaps):
        self.snap_iterations = np.ascontiguousarray(
          np.linspace( *(it_range+(nSnaps+1, )), dtype=np.uint32)[1:])

        self.snap_iterations = arrcl.to_device(self.queue, self.snap_iterations)


    def _set_global_working_group_size(self):
        # self.WGS = self.ctx.devices[0].max_work_group_size
        if self.dev_type=='CPU':
            self.WGS = 32
        else:
            self.WGS = 256

    def _get_wgs(self, Nelem):
        if Nelem <= self.WGS:
            return Nelem, Nelem
        else:
            WGS_tot = int(np.ceil(1.*Nelem/self.WGS))*self.WGS
            WGS = self.WGS
            return WGS, WGS_tot

    def _read_args(self, file_spectrum):
        if self.rank == 0:
            self.Data = {}
            self.Data['radiation'] = {}
            self.Args = {}
            self.Args['Features'] = {}

            with h5py.File(file_spectrum, "r") as f:
                for key in f['radiation'].keys():
                    self.Data['radiation'][key] = f['radiation/'+key][()]

                ArgsKeys = list(f['Args'].keys())
                if 'Features' in ArgsKeys:
                    ArgsKeys.remove('Features')
                    for key in f['Args/Features'].keys():
                        self.Args['Features'][key] = f['Args/Features'+key][()]

                for key in ArgsKeys:
                    self.Args[key] = f['Args/'+key][()]

                self.snap_iterations = f['snap_iterations'][()]
                self.total_weight = f['total_weight'][()]