import numpy as np
import h5py
import pyopencl as cl
#cl.create_some_context()
# TODO: remove commented lines in this file
import pyopencl.array as arrcl
from mako.template import Template

from tqdm import tqdm
from .utils import Utilities
from synchrad import __path__ as src_path

try:
    from mpi4py import MPI
    mpi_installed = True
except (ImportError):
    mpi_installed = False

src_path = src_path[0] + '/'


class SynchRad(Utilities):
    """
    Main class of SynchRad which contains all methods for construction
    of SR calculator object, running calculation, importing track data
    and exporting the radiation data. Inheritesmethods to analyze the
    simulation results.
    """

    def __init__(self, Args={}, file_spectrum=None):
        """
        Initializes SynchRad using either a dictionary `Args` with calculation
        parameters or exporting simulation from the file `file_spectrum`.
        When initializing from file_spectrum` the file, should be created from
        an executed simulation by the `calculate_spectrum` method

        Arguments available in `Args` dictionary
        ----------
        grid: list
            List of parameters to construct a 3D grid. For the `far-field`
            calculations grid defines the spherical coordinates
            `(omega, theta, phi)`, where frequency omega is in the units of
            `2*pi*c/lambda_u` with `lambda_u` being unit distance used for 
            tracked corrdinates, and `theta` and `phi` are the elevation and
            rotation angles in radians. For the `near-field` calculations,
            elevation angle `theta` is replaced by the radius `R` in the units
            of coordinates.
            Format for the far-field:
            "grid": [
                      (omega_min, omega_max),
                      (theta_min, theta_max),
                      (phi_min, phi_max),
                      (N_omega, N_theta, N_phi),
                    ]
            Format for the near-field:
            "grid": [
                      (omega_min, omega_max),
                      (R_min, R_max),
                      (phi_min, phi_max),
                      (N_omega, N_R, N_phi),
                    ]

        mode: string (optional)
            Calculation type which is by default far-field `"far"` or can
            be near-field `"near"`.

        dtype: string (optional)
            Numerical precision to be used. By default is "`double`", but
            can be set to "`single`" (be careful with that though).

        ctx: string (optional)
            Define of openCL context. If not provided will provide this
            choice interactively (or take default if only one is available).
            Possible choices are:
            'none': initialized without any device
            'mpi': use platform `0` and map multiple avaliable devices vie MPI
            list of choices for platform and device [PlatformID, DeviceID]

        Features: list of strings (optional)
            Additional features. Currently has following options:
              'wavelengthGrid': make frequency axis with uniform 
                wavelength intervals 
              'logGrid': make frequency axis with intervals growing 
                logarithmically
        """

        if mpi_installed:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
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
                           timeStep=None, comp='total', L_screen=None,
                           Np_max=None, it_range=None,
                           nSnaps=1, sigma_particle=0,
                           weights_normalize=None,
                           file_spectrum=None,
                           verbose=True):
        """
        Main method to run the SR calculation.

        Parameters
        ----------
        particleTracks: list
            Container of the particles tracks. Each track has the format:
            [x, y, z, uz, uy, uz, w, it_start], where first 6 are the
            numpy arrays of coordinates and normalized momenta (`beta*gamma`),
            `w` is a particle weight which defines a number of the real
            particles, and `it_start` is an time step at which particle appears
            relative to the whole interaction.

        file_tracks: string
            Path to the h5-file with tracks in a specific format, as created
            by the available convertsion utilities (openPMD, VSIM)

        timeStep: double
            Step used in the tracks defined as `c*dt` and in the same units of
            distance as the coordinates

        L_screen: double
            For the near-field calculations, the distance to the screen should
            be defined. Has the same units as the coordinates.

        comp: string (optional)
            Define which vector components of emitted light to calculate.
            Available choices are:
              'total' (default): sum incoherently all components:
                `SUM_tracks( |A_x|^2 + |A_y|^2 + |A_z|^2)`
              'cartesian': record Cartesian components incoherently:
                `SUM_tracks(|A_x|^2), SUM_tracks(|A_y|^2), SUM_tracks(|A_z|^2)`
              'spheric': record spheric components incoherently 
                (far-field only): `SUM_tracks(|A_r|^2), 
                SUM_tracks(|A_theta|^2), SUM_tracks(|A_phi|^2)`
              'cartesian_complex': record Cartesian components coherently:
                `SUM_tracks(A_x), SUM_tracks(A_y), SUM_tracks(A_z)`

        sigma_particle : double (optional) 
            Define size of the particle in distance units with Gaussian form-factor

        weights_normalize : string or None (optional)
            Reset the particle weights with some normalization (needed for the 
            coherency effects with macroparticles). Can be 'mean', 'max' or 'ones' 
            to normalize weights with a mean or max weight (over all 
            tracks) or set them to ones respectively.

        Np_max : integer
            Define a number of tracks to use in calculation

        nSnaps : integer (optional)
            Number of records to make along the interaction time with uniform intervals

        it_range :
            Specify the range of iterations to consider along the interaction

        file_spectrum : string
            Path and name to the file to which write the radiation data along with 
            the simulation configuration
        """

        self.Args['sigma_particle'] = self.dtype(sigma_particle)
        if self.Args['mode'] == 'near':
            if L_screen is not None:
                self.Args['L_screen'] = L_screen
            else:
                print('Define L_screen argument for near-field calculation')

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
        self.total_weight = 0.0

        if weights_normalize=='mean' or weights_normalize=='max':
            weights = []

            for track in particleTracks:
                weights.append(track[6])

            if weights_normalize=='mean':
                weight_norm = np.mean(weights)
            elif weights_normalize=='max':
                weight_norm = np.max(weights)

        if self.rank==0:
            calc_iterator = tqdm(range(len(particleTracks)))
        else:
            calc_iterator = range(len(particleTracks))

        for itr in calc_iterator:
            track = particleTracks[itr]

            if weights_normalize=='mean' or weights_normalize=='max':
                track[6] /= weight_norm
            elif weights_normalize=='ones':
                track[6] = 1.0

            self.total_weight += track[6]
            track = self._track_to_device(track)
            self._process_track(track, comp, nSnaps, it_range)

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
                for key in ArgsKeys:
                    f_out['Args/'+key] = self.Args[key]

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

        if self.Args['mode'] == 'far':
            axs_str = ('omega', 'sinTheta', 'cosTheta', 'sinPhi', 'cosPhi')
        elif self.Args['mode'] == 'near':
            axs_str = ('omega', 'radius', 'sinPhi', 'cosPhi')

        args_axes = [self.Data[name].data for name in axs_str]

        if self.Args['mode'] == 'near':
            args_axes += [ self.dtype(self.Args['L_screen']), ]

        args_res = [np.uint32(Nn) for Nn in self.Args['gridNodeNums']]
        args_aux = [self.Args['timeStep'], nSnaps, self.snap_iterations.data]

        args = args_track + args_axes + args_res + args_aux

        if comp == 'total':
            self._mapper.total(self.queue, (WGS_tot, ), (WGS, ),
                               spect['total'].data, *args)
        elif comp == 'cartesian':
            self._mapper.cartesian_comps(self.queue, (WGS_tot, ), (WGS, ),
                spect['x'].data, spect['y'].data, spect['z'].data, *args)

        elif comp == 'cartesian_complex':
            arg_FormFactor = [self.Data['FormFactor'].data,]
            args += arg_FormFactor
            self._mapper.cartesian_comps_complex(
                self.queue, (WGS_tot, ), (WGS, ),
                spect['xre'].data, spect['xim'].data,
                spect['yre'].data, spect['yim'].data,
                spect['zre'].data, spect['zim'].data,
                *args)

        elif comp == 'spheric':
            self._mapper.spheric_comps(self.queue, (WGS_tot, ), (WGS, ),
                spect['r'].data, spect['theta'].data, spect['phi'].data, *args)

        elif comp == 'spheric_complex':
            arg_FormFactor = [self.Data['FormFactor'].data,]
            args += arg_FormFactor
            self._mapper.spheric_comps_complex(
                self.queue, (WGS_tot, ), (WGS, ),
                spect['rre'].data, spect['rim'].data,
                spect['thetare'].data, spect['thetaim'].data,
                spect['phire'].data, spect['phiim'].data,
                *args)

    def _init_args(self, Args):
        self.Args = Args

        if 'mode' not in self.Args.keys():
            self.Args['mode'] = 'far'

        if 'dtype' not in self.Args:
            self.Args['dtype'] = 'double'

        if self.Args['dtype'] == 'double':
            self.dtype = np.double
        elif self.Args['dtype'] == 'float':
            self.dtype = np.single

        if self.Args['dtype'] == 'float':
            if self.Args['mode'] == 'far':
                print ( 'WARNING: Chosen single precision should be ' + \
                        'used with care for the farfield calculations\n' )
            elif self.Args['mode'] == 'near':
                print ( 'WARNING: Chosen single precision is not ' + \
                        'recommended for the nearfield calculations\n' )

        if 'ctx' not in self.Args.keys():
            self.Args['ctx'] = None

        self.Args['gridNodeNums'] = self.Args['grid'][-1]
        self.Args['numGridNodes'] = int(np.prod(self.Args['gridNodeNums']))

        No = self.Args['gridNodeNums'][0]
        omega_min, omega_max = self.Args['grid'][0]

        if 'Features' not in self.Args.keys():
            self.Args['Features'] = []

        omega = np.r_[omega_min:omega_max:No*1j]
        for feature in self.Args['Features']:
            if feature == 'wavelengthGrid':
                self.Args['wavelengths'] = np.r_[
                    1. / omega_max : 1. / omega_min : No * 1j ]
                omega = 1./self.Args['wavelengths']
                break
            elif feature == 'logGrid':
                d_log_w = np.log(omega_max / omega_min) / (No - 1.0)
                omega = omega_min * np.exp( d_log_w * np.arange(No) )
                break

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
            Nr, Np = self.Args['gridNodeNums'][1:]

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
                     'cartesian_complex':['xre', 'xim','yre',
                                          'yim', 'zre', 'zim'],
                     'spheric':['r', 'theta', 'phi'],
                     'spheric_complex':['rre', 'rim','thetare',
                                        'thetaim', 'phire', 'phiim'],
                     'total': ['total',]
                     }

        self.Args['comp'] = comp

        if self.Args['mode'] == 'near':
            self.Args['theta']  = np.arctan2( self.Args['radius'],
                                              self.Args['L_screen'] )

        self.Data['radiation'] = {}

        exp_factor = self.dtype(-0.5) * \
            (self.dtype(2*np.pi) * self.Args['omega'] * \
             self.Args['sigma_particle'])**2

        self.Data['FormFactor'] = arrcl.to_device( self.queue,
                                                   np.exp( exp_factor ) )

        for vec_comp in vec_comps[comp]:
            self.Data['radiation'][vec_comp] = \
                arrcl.zeros(self.queue, radiation_shape, dtype=self.dtype)

    def _init_data(self):

        self.Data = {}

        if self.plat_name == "None":
            return

        # Note that dw is not multiplied by 2*pi
        self.Data['omega'] = arrcl.to_device( self.queue,
                                 self.dtype(2*np.pi) * self.Args['omega'] )

        if self.Args['mode'] == 'far':
            self.Data['sinTheta'] = arrcl.to_device( self.queue,
                                                 np.sin(self.Args['theta']) )
            self.Data['cosTheta'] = arrcl.to_device( self.queue,
                                                 np.cos(self.Args['theta']) )
            self.Data['sinPhi'] = arrcl.to_device( self.queue,
                                                   np.sin(self.Args['phi']) )
            self.Data['cosPhi'] = arrcl.to_device( self.queue,
                                                   np.cos(self.Args['phi']) )
        elif self.Args['mode'] == 'near':
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
        elif self.Args['ctx'] == 'mpi':
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
                print(f"Creating context with args: {ctx_kw_args}")  # Logging

                # Set up OpenCL context
                platforms = cl.get_platforms()
                gpus = platforms[0].get_devices(device_type=cl.device_type.GPU)

                # Map MPI rank to a specific GPU
                device = gpus[self.rank % len(gpus)]
                self.ctx = cl.Context(devices=[device])
                self.queue = cl.CommandQueue(self.ctx)

                #self.ctx = cl.create_some_context(**ctx_kw_args)
                #self.queue = cl.CommandQueue(self.ctx)

                selected_dev = self.queue.device
                self.dev_type = cl.device_type.to_string(selected_dev.type)
                self.dev_name = self.queue.device.name

                self.plat_name = selected_dev.platform.vendor
                self.ocl_version = selected_dev.opencl_c_version
                print(f"Context created successfully on device: {self.dev_name}")  # Logging
            except Exception as e:
                print(f"Failed to create context: {e}")  # Error logging
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
            self.Data['radiation'][key] = np.ascontiguousarray(buff,
                                            dtype=np.double)

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

        if self.plat_name == "None":
            return

        agrs = {}
        agrs['my_dtype'] = self.Args['dtype']
        if 'native' in self.Args:
            agrs['f_native'] = 'native_'
        else:
            agrs['f_native'] = ''

        fname = src_path
        if self.Args['mode'] == 'far':
            fname += "kernel_farfield.cl"
        elif self.Args['mode'] == 'near':
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

            with h5py.File(file_spectrum, "r") as f:
                for key in f['radiation'].keys():
                    self.Data['radiation'][key] = f['radiation/'+key][()]

                for key in f['Args'].keys():
                    self.Args[key] = f['Args/'+key][()]

                    # in h5py >=3.0, the strings in datasets are bytes!
                    if isinstance(self.Args[key], bytes):
                        self.Args[key] = self.Args[key].decode()

                self.snap_iterations = f['snap_iterations'][()]
                self.total_weight = f['total_weight'][()]
