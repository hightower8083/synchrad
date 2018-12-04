import numpy as np

import pyopencl as cl
import pyopencl.array as arrcl
from mako.template import Template

from .utils import Utilities

from synchrad import __path__ as src_path
src_path = src_path[0] + '/'


class SynchRad(Utilities):

    def __init__(self, Args={}):

        self._init_args(Args)
        self._init_comm()
        self._init_data()
        self._compile_kernels()

    def calculate_spectrum(self, particleTracks, comp='all'):
        if self.Args['mode'] == 'far':
            for particleTrack in particleTracks:
                particleTrack = self._track_to_device(particleTrack)
                self._process_track( particleTrack, comp=comp )
            self._spectr_from_device()
        else:
            raise NotImplementedError('Only far-field is available for now')

    def _spectr_from_device(self):
        self.Data['radiation'] = self.Data['radiation'].get().T

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

    def _process_track(self, particleTrack, comp, return_event=False):

        compDict = {'x':0, 'y':1, 'z':2}

        x, y, z, ux, uy, uz, wp = particleTrack
        spect = self.Data['radiation']
        WGS, WGS_tot = self._get_wgs(self.Args['numGridNodes'])
        No, Nt, Np = self.Args['grid'][-1]

        args_track = [coord.data for coord in (x, y, z, ux, uy, uz)]
        args_track += [ wp, np.uint32(x.size) ]

        args_axes = []
        for name in ('omega', 'cosTheta', 'sinTheta', 'cosPhi', 'sinPhi'):
            args_axes.append( self.Data[name].data )
        args_res = [np.uint32(Nn) for Nn in (No, Nt, Np)]
        args_aux = [self.dtype(self.Args['timeStep']), ]

        args = args_track + args_axes + args_res + args_aux
        if comp is 'all':
            evnt = self._farfield.total( self.queue, (WGS_tot, ), (WGS, ),
                                         spect.data, *args )
        else:
            args = [ np.uint32(compDict[comp]), ] + args
            evnt = self._farfield.single_component( self.queue, (WGS_tot, ),
                                                 (WGS, ), spect.data, *args )

        if return_event:
            return evnt
        else:
            evnt.wait()

    def _compile_kernels(self):

        agrs = {'my_dtype': self.Args['dtype'], 'f_native':self.f_native}
        fname_far = src_path + "kernel_farfield.cl"

        src_far = Template( filename=fname_far ).render(**agrs)

        compiler_options = []
        # can be set to  `['-cl-fast-relaxed-math',]` but is probably
        # usless in most cases
        self._farfield = cl.Program(self.ctx, src_far)\
            .build(options=compiler_options)

    def _set_global_working_group_size(self):
        if self.dev_type=='CPU':
            self.WGS = 32
        else:
            self.WGS = 256
            # should be `self.ctx.devices[0].max_work_group_size`, but
            # fails for some implementations

    def _get_wgs(self, Nelem):
        if Nelem <= self.WGS:
            return Nelem, Nelem
        else:
            WGS_tot = int(np.ceil(1.*Nelem/self.WGS))*self.WGS
            WGS = self.WGS
            return WGS, WGS_tot

    def _init_args(self, Args):
        self.Args = Args

        if 'dtype' not in self.Args:
            self.Args['dtype'] = 'float'

        if self.Args['dtype'] is 'float':
            self.dtype = np.single
        elif self.Args['dtype'] is 'double':
            self.dtype = np.double

        if 'no_native' in self.Args:
            self.f_native = ''
        else:
            self.f_native = 'native_'

        if 'timeStep' not in self.Args.keys():
            raise KeyError("timeStep must be defined.")

        if 'ctx' not in self.Args.keys():
            self.Args['ctx'] = None

        if 'Features' not in self.Args.keys():
            self.Args['Features'] = {}

        if 'mode' not in self.Args.keys():
            self.Args['mode'] = 'far'
        elif self.Args['mode'] != 'far':
            raise NotImplementedError('Only far-field is available for now')

        omega_min, omega_max = self.Args['grid'][0]
        No, Nt, Np = self.Args['grid'][-1]

        self.Args['numGridNodes'] = No * Nt * Np

        if 'wavelengthGrid' in self.Args['Features']:
            self.Args['wavelengths'] = np.r_[1./omega_max:1./omega_min:No*1j]
            omega = 1./self.Args['wavelengths']
        else:
            omega = np.r_[omega_min:omega_max:No*1j]

        self.Args['omega'] = omega.astype(self.dtype)

        if No>1:
            self.Args['dw'] = np.abs( omega[1:]-omega[:-1] )
        else:
            self.Args['dw'] = np.array([1.,], dtype=self.dtype)

        if self.Args['mode'] == 'far':
            theta_min, theta_max  = self.Args['grid'][1]
            phi_min, phi_max = self.Args['grid'][2]

            theta = np.r_[theta_min:theta_max:Nt*1j]
            phi = phi_min + (phi_max-phi_min)/Np*np.arange(Np)

            self.Args['theta'] = theta.astype(self.dtype)
            self.Args['phi'] = phi.astype(self.dtype)

            if Nt>1:
                self.Args['dth'] = theta[1] - theta[0]
            else:
                self.Args['dth'] = self.dtype(1.)

            if Np>1:
                self.Args['dph'] = phi[1] - phi[0]
            else:
                self.Args['dph'] = self.dtype(1.)

            self.Args['dV'] = self.Args['dw']*self.Args['dth']*self.Args['dph']


    def reinit(self):
        No, Nt, Np = self.Args['grid'][-1]
        self.Data['radiation'] = arrcl.zeros( self.queue, (Np, Nt, No),
                                        dtype=self.dtype )
    def _init_data(self):

        self.Data = {}

        No, Nt, Np = self.Args['grid'][-1]
        self.Data['radiation'] = arrcl.zeros( self.queue, (Np, Nt, No),
                                                dtype=self.dtype )

        self.Data['omega'] = arrcl.to_device( self.queue,
                                     2*np.pi*self.Args['omega'] )

        if self.Args['mode'] == 'far':
            self.Data['cosTheta'] = arrcl.to_device( self.queue,
                                        np.cos(self.Args['theta']) )
            self.Data['sinTheta'] = arrcl.to_device( self.queue,
                                        np.sin(self.Args['theta']) )
            self.Data['cosPhi'] = arrcl.to_device( self.queue,
                                        np.cos(self.Args['phi']) )
            self.Data['sinPhi'] = arrcl.to_device( self.queue,
                                        np.sin(self.Args['phi']) )

    def _init_comm(self):

        ctx_kw_args = {}
        if self.Args['ctx'] is None:
            ctx_kw_args['interactive'] = True
        else:
            ctx_kw_args['answers'] = self.Args['ctx']

        self.ctx = cl.create_some_context(**ctx_kw_args)
        self.queue = cl.CommandQueue(self.ctx)

        selected_dev = self.queue.device
        self.dev_type = cl.device_type.to_string(selected_dev.type)
        self.dev_name = self.queue.device.name

        self.plat_name = selected_dev.platform.vendor
        self.ocl_version = selected_dev.opencl_c_version

        print("{} device: {} \nPlatform: {}\nCompiler: {}".\
               format( self.dev_type, self.dev_name,
                       self.plat_name, self.ocl_version) )

        self._set_global_working_group_size()
