// kernels of far field calculation (total and single component)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void total(
  __global ${my_dtype} *spectrum,
  __global ${my_dtype} *x,
  __global ${my_dtype} *y,
  __global ${my_dtype} *z,
  __global ${my_dtype} *ux,
  __global ${my_dtype} *uy,
  __global ${my_dtype} *uz,
           ${my_dtype} wp,
                  uint nSteps,
  __global ${my_dtype} *omega,
  __global ${my_dtype} *radius,
  __global ${my_dtype} *sinPhi,
  __global ${my_dtype} *cosPhi,
           ${my_dtype} distanceToScreen,
                  uint nOmega,
                  uint nRadius,
                  uint nPhi,
           ${my_dtype} dt )
{
  uint gti = (uint) get_global_id(0);

  if (gti < nRadius*nPhi*nOmega)
   {

    uint iPhi = gti / (nOmega * nRadius);
    uint iRadius = (gti - iPhi*nOmega*nRadius) / nOmega;
    uint iOmega = gti - iPhi*nOmega*nRadius - iRadius*nOmega;

    ${my_dtype} omegaLocal = omega[iOmega];

    ${my_dtype}3 coordOnScreen = (${my_dtype}3) { distanceToScreen,
                                                  radius[iRadius]*sinPhi[iPhi],
                                                  radius[iRadius]*cosPhi[iPhi] };

    ${my_dtype}3 nVec, uLocal, uNextLocal;
    ${my_dtype}3 spectrLocalRe = (${my_dtype}3) {0., 0., 0.};
    ${my_dtype}3 spectrLocalIm = (${my_dtype}3) {0., 0., 0.};

    ${my_dtype} time, phase, dPhase, sinPhase, cosPhase, r, rInv, gammaInv;

    ${my_dtype} wpdt2 = wp*dt*dt;
    ${my_dtype} phasePrev = ${my_dtype}(0.);

    for (uint it=0; it<nSteps-1; it++){

      time = ${my_dtype}(it * dt);
      nVec = coordOnScreen;
      nVec -= (${my_dtype}3) {x[it], y[it], z[it]};
      r = length(nVec);

      phase = omegaLocal * (time + r) ;
      dPhase = fabs(phase - phasePrev);
      phasePrev = phase;

      if ( dPhase < ${my_dtype}(M_PI) ) {

        rInv = ${my_dtype}(1.0)/r;
        nVec *= rInv;

        uLocal = (${my_dtype}3) {ux[it], uy[it], uz[it]};
        // for the staggered X and U
        uNextLocal = (${my_dtype}3) {ux[it+1], uy[it+1], uz[it+1]};
        uLocal = ${my_dtype}(0.5) * (uNextLocal+uLocal);

        gammaInv = ${f_native}rsqrt( ${my_dtype}(1.) + dot(uLocal, uLocal) );
        uLocal *= gammaInv;

        sinPhase = ${f_native}sin(phase);
        cosPhase = ${f_native}cos(phase);

        // re-using local variables
        uLocal = (uLocal - nVec) * omegaLocal * rInv;
        nVec *= rInv * rInv;

        spectrLocalRe += nVec*cosPhase - uLocal*sinPhase;
        spectrLocalIm += nVec*sinPhase + uLocal*cosPhase;
      }
    }

    spectrum[gti] +=  wpdt2 * ( dot(spectrLocalRe, spectrLocalRe)
                              + dot(spectrLocalIm, spectrLocalIm) );
   }
}
