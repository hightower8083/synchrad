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
  __global ${my_dtype} *cosTheta,
  __global ${my_dtype} *sinTheta,
  __global ${my_dtype} *cosPhi,
  __global ${my_dtype} *sinPhi,
             uint nOmega,
             uint nTheta,
             uint nPhi,
           ${my_dtype} dt )
{
  uint gti = (uint) get_global_id(0);

  if (gti < nTheta*nPhi*nOmega)
   {

    uint iPhi = gti / (nOmega * nTheta);
    uint iTheta = (gti - iPhi*nOmega*nTheta) / nOmega;
    uint iOmega = gti - iPhi*nOmega*nTheta - iTheta*nOmega;

    ${my_dtype} omegaLocal = omega[iOmega];
    ${my_dtype}3 nVec = (${my_dtype}3) { cosTheta[iTheta],
                       sinTheta[iTheta]*sinPhi[iPhi],
                       sinTheta[iTheta]*cosPhi[iPhi] };


    ${my_dtype}3 xLocal;
    ${my_dtype}3 uLocal;
    ${my_dtype}3 uNextLocal;
    ${my_dtype}3 aLocal;
    ${my_dtype}3 amplitude;

    ${my_dtype} time, phase, dPhase, sinPhase, cosPhase;
    ${my_dtype} c1, c2, gammaInv;

    ${my_dtype} dtInv = (${my_dtype}) 1./dt;
    ${my_dtype} wpdt2 =(${my_dtype}) wp*dt*dt;
    ${my_dtype} phasePrev = (${my_dtype}) 0.0;
    ${my_dtype}3 spectrLocalRe = (${my_dtype}3) {0., 0., 0.};
    ${my_dtype}3 spectrLocalIm = (${my_dtype}3) {0., 0., 0.};

    for (uint it=0; it<nSteps-1; it++){

      time = (${my_dtype}) (it * dt);
      xLocal = (${my_dtype}3) {x[it], y[it], z[it]};

      phase = omegaLocal * ( time - dot(xLocal, nVec)) ;
      dPhase = fabs(phase - phasePrev);
      phasePrev = phase;

      if ( dPhase < (${my_dtype}) M_PI) {

        uLocal = (${my_dtype}3) {ux[it], uy[it], uz[it]};
        uNextLocal = (${my_dtype}3) {ux[it+1], uy[it+1], uz[it+1]};

        gammaInv = ${f_native}rsqrt( ${my_dtype}(1.) + dot(uLocal, uLocal) );
        uLocal *= gammaInv;
        gammaInv = ${f_native}rsqrt( ${my_dtype}(1.) + dot(uNextLocal, uNextLocal) );
        uNextLocal *= gammaInv;

        aLocal = (uNextLocal-uLocal) * dtInv;
        uLocal = ${my_dtype}(0.5) * (uNextLocal+uLocal);

        c1 = dot(aLocal, nVec);
        c2 = ${my_dtype}(1.) - dot(uLocal, nVec);

        c2 =  ${my_dtype}(1.) / c2;
        c1 = c1*c2*c2;

        sinPhase = ${f_native}sin(phase);
        cosPhase = ${f_native}cos(phase);

        amplitude = c1*( nVec-uLocal ) - c2*aLocal;
        spectrLocalRe += amplitude* cosPhase;
        spectrLocalIm += amplitude* sinPhase;
      }
    }

    spectrum[gti] +=  wpdt2 * ( dot(spectrLocalRe, spectrLocalRe) + dot(spectrLocalIm, spectrLocalIm) );
   }
}

__kernel void single_component(
  __global ${my_dtype} *spectrum,
             uint iComponent,
  __global ${my_dtype} *x,
  __global ${my_dtype} *y,
  __global ${my_dtype} *z,
  __global ${my_dtype} *ux,
  __global ${my_dtype} *uy,
  __global ${my_dtype} *uz,
           ${my_dtype} wp,
             uint nSteps,
  __global ${my_dtype} *omega,
  __global ${my_dtype} *cosTheta,
  __global ${my_dtype} *sinTheta,
  __global ${my_dtype} *cosPhi,
  __global ${my_dtype} *sinPhi,
             uint nOmega,
             uint nTheta,
             uint nPhi,
           ${my_dtype} dt )
{
  uint gti = (uint) get_global_id(0);

  if (gti < nTheta*nPhi*nOmega)
   {

    uint iPhi = gti / (nOmega * nTheta);
    uint iTheta = (gti - iPhi*nOmega*nTheta) / nOmega;
    uint iOmega = gti - iPhi*nOmega*nTheta - iTheta*nOmega;

    ${my_dtype} omegaLocal = omega[iOmega];
    ${my_dtype}3 nVec = (${my_dtype}3) { cosTheta[iTheta],
                       sinTheta[iTheta]*sinPhi[iPhi],
                       sinTheta[iTheta]*cosPhi[iPhi] };


    ${my_dtype}3 xLocal;
    ${my_dtype}3 uLocal;
    ${my_dtype}3 uNextLocal;
    ${my_dtype}3 aLocal;
    ${my_dtype} amplitude;

    ${my_dtype} time, phase, dPhase, sinPhase, cosPhase;
    ${my_dtype} c1, c2, gammaInv;

    ${my_dtype} dtInv = 1./dt;
    ${my_dtype} wpdt2 = wp*dt*dt;
    ${my_dtype} phasePrev = 0;
    ${my_dtype} spectrLocalRe = (${my_dtype}) 0.0;
    ${my_dtype} spectrLocalIm = (${my_dtype}) 0.0;

    for (uint it=0; it<nSteps-1; it++){

      time = it * dt;
      xLocal = (${my_dtype}3) {x[it], y[it], z[it]};

      phase = omegaLocal * ( time - dot(xLocal, nVec)) ;
      dPhase = fabs(phase - phasePrev);
      phasePrev = phase;

      if ( dPhase < (${my_dtype}) M_PI) {

        uLocal = (${my_dtype}3) {ux[it], uy[it], uz[it]};
        uNextLocal = (${my_dtype}3) {ux[it+1], uy[it+1], uz[it+1]};

        gammaInv = ${f_native}rsqrt( ${my_dtype}(1.) + dot(uLocal, uLocal) );
        uLocal *= gammaInv;
        gammaInv = ${f_native}rsqrt( ${my_dtype}(1.) + dot(uNextLocal, uNextLocal) );
        uNextLocal *= gammaInv;

        aLocal = (uNextLocal-uLocal) * dtInv;
        uLocal = ${my_dtype}(0.5) * (uNextLocal+uLocal);

        c1 = dot(aLocal, nVec);
        c2 = ${my_dtype}(1.) - dot(uLocal, nVec);

        c2 = ${my_dtype}(1.)/c2;
        c1 = c1*c2*c2;

        sinPhase = ${f_native}sin(phase);
        cosPhase = ${f_native}cos(phase);

        amplitude = c1*( nVec[iComponent]-uLocal[iComponent] ) - c2*aLocal[iComponent];
        spectrLocalRe += amplitude* cosPhase;
        spectrLocalIm += amplitude* sinPhase;
      }
    }

    spectrum[gti] +=  wpdt2 * ( spectrLocalRe*spectrLocalRe + spectrLocalIm*spectrLocalIm );
   }
}
