import numpy as np
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

    def __init__(self, Args={}):

        if mpi_installed:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.rank
            self.size = self.comm.size
        else:
            self.rank = 0
            self.size = 1

        self._init_args(Args)
        self._init_comm()
        self._init_data()
        self._compile_kernels()

    def calculate_spectrum( self, particleTracks=[],
                            h5_file=None, comp='total',
                            Np_max=None, nSnaps=1,
                            verbose=True ):

        self._init_raditaion(comp, nSnaps)

        if h5_file is not None:
            particleTracks=[]

            if self.rank==0:
                print('Input from the file, list input is ignored')

            Np = h5_file['misc/N_particles'][()]
            cmps = ('x', 'y', 'z', 'ux', 'uy', 'uz', 'w')
            if Np_max is not None:
                Np = min(Np_max, Np)

            # load all tracks to RAM (need more speed tests)
            part_ind = np.arange(Np)[self.rank::self.size]
            for ip in part_ind:
                track = [h5_file[f'tracks/{ip:d}/{cmp}'][()] for cmp in cmps]
                particleTracks.append(track)

            if self.rank==0:
                print('Tracks are loaded')

            itr = 0
            for track in particleTracks:
                track = self._track_to_device(track)
                self._process_track(track, comp=comp)
                itr += 1
                if self.rank==0 and verbose:
                    progress = itr/len(particleTracks) * 100
                    print("Done {:0.1f}%".format(progress),
                          end='\r', flush=True)
        else:
            Np = len(particleTracks)
            if Np_max is not None:
                Np = min(Np_max, Np)

            particleTracks = particleTracks[:Np][self.rank::self.size]

            itr = 0
            for track in particleTracks:
                track = self._track_to_device(track)
                self._process_track(track, comp, nSnaps)
                itr += 1
                if self.rank==0 and verbose:
                    progress = itr/len(particleTracks) * 100
                    print("Done {:0.1f}%".format(progress),
                          end='\r', flush=True)

        self._spectr_from_device()
        if mpi_installed:
            self._gather_result_mpi()

    def _gather_result_mpi(self):
        buff = np.zeros_like(self.Data['radiation'])
        self.comm.barrier()
        self.comm.Reduce([self.Data['radiation'], MPI.DOUBLE],
                         [buff, MPI.DOUBLE])
        self.comm.barrier()
        self.Data['radiation'] = buff

    def _spectr_from_device(self):
        buff = self.Data['radiation'].get()
        if len(buff.shape)>3:
            buff = buff.swapaxes(-1,1)
        else:
            buff = buff.T
        self.Data['radiation'] = np.ascontiguousarray(buff, dtype=np.double)

    def _track_to_device(self, particleTrack):
        x, y, z, ux, uy, uz, wp = particleTrack
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
        particleTrack = [x, y, z, ux, uy, uz, wp]
        return particleTrack

    def _process_track(self, particleTrack, comp, nSnaps,
                       return_event=False):

        x, y, z, ux, uy, uz, wp = particleTrack
        spect = self.Data['radiation']
        WGS, WGS_tot = self._get_wgs(self.Args['numGridNodes'])

        args_track = [coord.data for coord in (x, y, z, ux, uy, uz)]
        args_track += [ wp, np.uint32(x.size) ]

        if self.Args['mode'] is 'far':
            axs_str = ('omega', 'sinTheta', 'cosTheta', 'sinPhi', 'cosPhi')
        elif self.Args['mode'] is 'near':
            axs_str = ('omega', 'radius', 'sinPhi', 'cosPhi')

        args_axes = [self.Data[name].data for name in axs_str]

        if self.Args['mode'] is 'near':
            args_axes += [ self.dtype(self.Args['grid'][3]), ]

        args_res = [np.uint32(Nn) for Nn in self.Args['grid'][-1]]
        args_aux = [self.dtype(self.Args['timeStep']), ]
        if 'total_series':
            args_aux += [np.uint32(nSnaps), ]

        args = args_track + args_axes + args_res + args_aux

        if comp is 'total':
            event = self._mapper.total( self.queue, (WGS_tot, ), (WGS, ),
                                         spect.data, *args )
        if comp is 'total_series':
            event = self._mapper.total_series( self.queue, (WGS_tot, ), (WGS, ),
                                         spect.data, *args )
        elif comp is 'cartesian':
            event = self._mapper.cartesian_comps( self.queue, (WGS_tot, ),
                                                  (WGS, ), spect.data, *args )
        elif comp is 'spheric':
            event = self._mapper.spheric_comps( self.queue, (WGS_tot, ),
                                                (WGS, ), spect.data, *args )

        if return_event:
            return event
        else:
            event.wait()

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

        if 'timeStep' not in self.Args.keys():
            raise KeyError("timeStep must be defined.")

        if 'ctx' not in self.Args.keys():
            self.Args['ctx'] = None

        if 'Features' not in self.Args.keys():
            self.Args['Features'] = {}

        gridNodeNums = self.Args['grid'][-1]

        self.Args['numGridNodes'] = int(np.prod(gridNodeNums))

        omega_min, omega_max = self.Args['grid'][0]
        No = gridNodeNums[0]
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
            Nt, Np = gridNodeNums[1:]
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

        if comp in ('cartesian', 'spheric'):
            all_comps=True
        else:
            all_comps=False

        radiation_shape = tuple(self.Args['grid'][-1][::-1])

        if all_comps:
            radiation_shape = (3,) + gridNodeNums

        if nSnaps > 1:
            radiation_shape = (nSnaps, ) + radiation_shape

        self.Data['radiation'] = arrcl.zeros( self.queue, radiation_shape,
                                              dtype=self.dtype )

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
