import numpy as np

import pyopencl as cl
import pyopencl.array as arrcl

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
        if self.Args['Mode'] == 'far':
            for particleTrack in particleTracks:
                particleTrack = self._track_to_device(particleTrack)
                self._process_track( particleTrack, comp=comp )
            self._spectr_from_device()
        else:
            raise NotImplementedError('Only far-field is available for now')

    def _spectr_from_device(self):
        self.Data['Rad'] = self.Data['Rad'].get().T

    def _track_to_device(self, particleTrack):
        x, y, z, ux, uy, uz, wp = particleTrack

        x = arrcl.to_device( self.queue, np.ascontiguousarray(x) )
        y = arrcl.to_device( self.queue, np.ascontiguousarray(y) )
        z = arrcl.to_device( self.queue, np.ascontiguousarray(z) )
        ux = arrcl.to_device( self.queue, np.ascontiguousarray(ux) )
        uy = arrcl.to_device( self.queue, np.ascontiguousarray(uy) )
        uz = arrcl.to_device( self.queue, np.ascontiguousarray(uz) )
        wp = np.double(wp)
        particleTrack = [x, y, z, ux, uy, uz, wp]
        return particleTrack

    def _process_track(self, particleTrack, comp, return_event=False):

        compDict = {'x':0, 'y':1, 'z':2}

        x, y, z, ux, uy, uz, wp = particleTrack
        spect = self.Data['Rad']
        WGS, WGS_tot = self._get_wgs(self.Args['numGridNodes'])
        No, Nt, Np = self.Args['Grid'][-1]

        args_track = [coord.data for coord in (x, y, z, ux, uy, uz)]
        args_track += [ wp, np.uint32(x.size) ]

        args_axes = []
        for name in ('omega', 'cosTheta', 'sinTheta', 'cosPhi', 'sinPhi'):
            args_axes.append( self.Data[name].data )
        args_res = [np.uint32(Nn) for Nn in (No, Nt, Np)]
        args_aux = [np.double(self.Args['TimeStep']), ]

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
        with open(src_path + "kernel_farfield.cl") as f:
            farfield_src = ''.join(f.readlines())

        self._farfield = cl.Program(self.ctx, farfield_src).build()

    def _set_global_working_group_size(self):
        if self.dev_type=='CPU':
            self.WGS = 32
        else:
            self.WGS = 256 #self.ctx.devices[0].max_work_group_size

    def _get_wgs(self, Nelem):
        if Nelem <= self.WGS:
            return Nelem, Nelem
        else:
            WGS_tot = int(np.ceil(1.*Nelem/self.WGS))*self.WGS
            WGS = self.WGS
            return WGS, WGS_tot

    def _init_args(self, Args):
        self.Args = Args

        if 'TimeStep' not in self.Args.keys():
            raise KeyError("TimeStep must be defined.")

        if 'ctx' not in self.Args.keys():
            self.Args['ctx'] = None

        if 'Features' not in self.Args.keys():
            self.Args['Features'] = {}

        if 'Mode' not in self.Args.keys():
            self.Args['Mode'] = 'far'
        elif self.Args['Mode'] != 'far':
            raise NotImplementedError('Only far-field is available for now')

        omega_min, omega_max = self.Args['Grid'][0]
        No, Nt, Np = self.Args['Grid'][-1]

        self.Args['numGridNodes'] = No * Nt * Np

        if 'wavelengthGrid' in self.Args['Features']:
            self.Args['wavelengths'] = np.r_[1./omega_max:1./omega_min:No*1j]
            omega = 1./self.Args['wavelengths']
        else:
            omega = np.r_[omega_min:omega_max:No*1j]
        self.Args['omega'] = omega

        if No>1:
            self.Args['dw'] = np.abs( omega[1:]-omega[:-1] )
        else:
            self.Args['dw'] = np.array([1.,])

        if self.Args['Mode'] == 'far':
            theta_min, theta_max  = self.Args['Grid'][1]
            phi_min, phi_max = self.Args['Grid'][2]

            theta = np.r_[theta_min:theta_max:Nt*1j]
            phi = phi_min + (phi_max-phi_min)/Np*np.arange(Np)

            self.Args['theta'] = theta
            self.Args['phi'] = phi

            if Nt>1:
                self.Args['dth'] = theta[1] - theta[0]
            else:
                self.Args['dth'] = 1.

            if Np>1:
                self.Args['dph'] = phi[1] - phi[0]
            else:
                self.Args['dph'] = 1.

            self.Args['dV'] = self.Args['dw']*self.Args['dth']*self.Args['dph']


    def reinit(self):
        No, Nt, Np = self.Args['Grid'][-1]
        self.Data['Rad'] = arrcl.zeros( self.queue, (Np, Nt, No),
                                                dtype=np.double )
    def _init_data(self):

        self.Data = {}

        No, Nt, Np = self.Args['Grid'][-1]
        self.Data['Rad'] = arrcl.zeros( self.queue, (Np, Nt, No),
                                                dtype=np.double )

        self.Data['omega'] = arrcl.to_device( self.queue,
                                              2*np.pi*self.Args['omega'] )

        if self.Args['Mode'] == 'far':
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

        print("{} DEVICE {} IS CHOSEN ON {} PLATFORM WITH {} COMPILER".\
               format( self.dev_type, self.dev_name,
                       self.plat_name, self.ocl_version) )

        self._set_global_working_group_size()
