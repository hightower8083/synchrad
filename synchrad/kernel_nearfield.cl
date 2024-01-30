// kernels of near field calculation (total and single component)

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
                  uint itStart,
                  uint itEnd,
                  uint nSteps,
  __global ${my_dtype} *omega,
  __global ${my_dtype} *radius,
  __global ${my_dtype} *sinPhi,
  __global ${my_dtype} *cosPhi,
           ${my_dtype} distanceToScreen,
                  uint nOmega,
                  uint nRadius,
                  uint nPhi,
           ${my_dtype} dt,
                  uint nSnaps,
  __global        uint *itSnaps )
{
  uint gti = (uint) get_global_id(0);
  uint nTotal = nRadius*nPhi*nOmega;

  if (gti < nTotal)
   {
    uint iPhi = gti / (nOmega * nRadius);
    uint iRadius = (gti - iPhi*nOmega*nRadius) / nOmega;
    uint iOmega = gti - iPhi*nOmega*nRadius - iRadius*nOmega;

    ${my_dtype} omegaLocal = omega[iOmega];

    ${my_dtype}3 coordOnScreen = (${my_dtype}3) { radius[iRadius]*cosPhi[iPhi],
                                                  radius[iRadius]*sinPhi[iPhi],
                                                  distanceToScreen };

    ${my_dtype}3 xLocal, uLocal, rVec, nVec, c1, c2;
    ${my_dtype} time, phase, dPhase, sinPhase, cosPhase, rLocal, rInv, gammaInv;

    ${my_dtype} wpdt2 =  wp * dt * dt;
    ${my_dtype} phasePrev = (${my_dtype}) 0.;
    ${my_dtype}3 spectrLocalRe = (${my_dtype}3) {0., 0., 0.};
    ${my_dtype}3 spectrLocalIm = (${my_dtype}3) {0., 0., 0.};

    uint iSnap, it_glob;
    for (iSnap=0; iSnap<nSnaps; iSnap++)
    {
      if (itStart < itSnaps[iSnap]) break;
    }

    for (uint it=0; it<itEnd-1; it++)
    {
      it_glob = itStart + it;

      if (it<nSteps-1)
      {
        time = (${my_dtype})it_glob * dt;
        xLocal = (${my_dtype}3) {x[it], y[it], z[it]};

        rVec = coordOnScreen - xLocal;
        rLocal = ${f_native}sqrt( dot(rVec, rVec) );

        phase = omegaLocal * (time + rLocal) ;
        dPhase = fabs(phase - phasePrev);
        phasePrev = phase;

        if ( dPhase < (${my_dtype})M_PI )
        {
          rInv = (${my_dtype})1. / rLocal;
          nVec = rInv * rVec;

          uLocal = (${my_dtype}3) {ux[it], uy[it], uz[it]};

          gammaInv = ${f_native}rsqrt( (${my_dtype})1. + dot(uLocal, uLocal) );
          uLocal *= gammaInv;

          sinPhase = ${f_native}sin(phase);
          cosPhase = ${f_native}cos(phase);

          c1 = omegaLocal * rInv * (uLocal - nVec);
          c2 = rInv * rInv * nVec;

          spectrLocalRe += -c1*sinPhase + c2*cosPhase;
          spectrLocalIm +=  c1*cosPhase + c2*sinPhase;
        }
      }

      if (it_glob+2 == itSnaps[iSnap])
      {
        spectrum[gti + nTotal*iSnap] +=  wpdt2 * (
          dot(spectrLocalRe, spectrLocalRe) +
          dot(spectrLocalIm, spectrLocalIm) );
        iSnap += 1;
      }
    }
  }
}

__kernel void cartesian_comps(
  __global ${my_dtype} *spectrum1,
  __global ${my_dtype} *spectrum2,
  __global ${my_dtype} *spectrum3,
  __global ${my_dtype} *x,
  __global ${my_dtype} *y,
  __global ${my_dtype} *z,
  __global ${my_dtype} *ux,
  __global ${my_dtype} *uy,
  __global ${my_dtype} *uz,
           ${my_dtype} wp,
                  uint itStart,
                  uint itEnd,
                  uint nSteps,
  __global ${my_dtype} *omega,
  __global ${my_dtype} *radius,
  __global ${my_dtype} *sinPhi,
  __global ${my_dtype} *cosPhi,
           ${my_dtype} distanceToScreen,
                  uint nOmega,
                  uint nRadius,
                  uint nPhi,
           ${my_dtype} dt,
                  uint nSnaps,
  __global        uint *itSnaps )
{
  uint gti = (uint) get_global_id(0);
  uint nTotal = nRadius*nPhi*nOmega;

  if (gti < nTotal)
   {
    uint iPhi = gti / (nOmega * nRadius);
    uint iRadius = (gti - iPhi*nOmega*nRadius) / nOmega;
    uint iOmega = gti - iPhi*nOmega*nRadius - iRadius*nOmega;

    ${my_dtype} omegaLocal = omega[iOmega];

    ${my_dtype}3 coordOnScreen = (${my_dtype}3) { radius[iRadius]*cosPhi[iPhi],
                                                  radius[iRadius]*sinPhi[iPhi],
                                                  distanceToScreen };

    ${my_dtype}3 xLocal, uLocal, rVec, nVec, c1, c2;
    ${my_dtype} time, phase, dPhase, sinPhase, cosPhase, rLocal, rInv, gammaInv;

    ${my_dtype} wpdt2 =  wp * dt * dt;
    ${my_dtype} phasePrev = (${my_dtype}) 0.;
    ${my_dtype}3 spectrLocalRe = (${my_dtype}3) {0., 0., 0.};
    ${my_dtype}3 spectrLocalIm = (${my_dtype}3) {0., 0., 0.};

    uint iSnap, it_glob;
    for (iSnap=0; iSnap<nSnaps; iSnap++)
    {
      if (itStart < itSnaps[iSnap]) break;
    }

    for (uint it=0; it<itEnd-1; it++)
    {
      it_glob = itStart + it;

      if (it<nSteps-1)
      {
        time = (${my_dtype})it_glob * dt;
        xLocal = (${my_dtype}3) {x[it], y[it], z[it]};

        rVec = coordOnScreen - xLocal;
        rLocal = ${f_native}sqrt( dot(rVec, rVec) );

        phase = omegaLocal * (time + rLocal) ;
        dPhase = fabs(phase - phasePrev);
        phasePrev = phase;

        if ( dPhase < (${my_dtype})M_PI )
        {
          rInv = (${my_dtype})1. / rLocal;
          nVec = rInv * rVec;

          uLocal = (${my_dtype}3) {ux[it], uy[it], uz[it]};
          gammaInv = ${f_native}rsqrt( (${my_dtype})1. + dot(uLocal, uLocal) );
          uLocal *= gammaInv;

          sinPhase = ${f_native}sin(phase);
          cosPhase = ${f_native}cos(phase);

          c1 = omegaLocal * rInv * (uLocal - nVec);
          c2 = rInv * rInv * nVec;

          spectrLocalRe += -c1*sinPhase + c2*cosPhase;
          spectrLocalIm +=  c1*cosPhase + c2*sinPhase;
        }
      }

      if (it_glob+2 == itSnaps[iSnap])
      {
        spectrum1[gti + nTotal*iSnap] +=  wpdt2 *
          (spectrLocalRe.s0*spectrLocalRe.s0 +
           spectrLocalIm.s0*spectrLocalIm.s0);

        spectrum2[gti + nTotal*iSnap] +=  wpdt2 *
          (spectrLocalRe.s1*spectrLocalRe.s1 +
           spectrLocalIm.s1*spectrLocalIm.s1);

        spectrum3[gti + nTotal*iSnap] +=  wpdt2 *
          (spectrLocalRe.s2*spectrLocalRe.s2 +
           spectrLocalIm.s2*spectrLocalIm.s2);
        iSnap += 1;
      }
    }
  }
}

__kernel void cartesian_comps_complex(
  __global ${my_dtype} *spectrum1_re,
  __global ${my_dtype} *spectrum1_im,
  __global ${my_dtype} *spectrum2_re,
  __global ${my_dtype} *spectrum2_im,
  __global ${my_dtype} *spectrum3_re,
  __global ${my_dtype} *spectrum3_im,
  __global ${my_dtype} *x,
  __global ${my_dtype} *y,
  __global ${my_dtype} *z,
  __global ${my_dtype} *ux,
  __global ${my_dtype} *uy,
  __global ${my_dtype} *uz,
           ${my_dtype} wp,
                  uint itStart,
                  uint itEnd,
                  uint nSteps,
  __global ${my_dtype} *omega,
  __global ${my_dtype} *radius,
  __global ${my_dtype} *sinPhi,
  __global ${my_dtype} *cosPhi,
           ${my_dtype} distanceToScreen,
                  uint nOmega,
                  uint nRadius,
                  uint nPhi,
           ${my_dtype} dt,
                  uint nSnaps,
  __global        uint *itSnaps,
  __global ${my_dtype} *FormFactor)
{
  uint gti = (uint) get_global_id(0);
  uint nTotal = nRadius*nPhi*nOmega;

  if (gti < nTotal)
   {
    uint iPhi = gti / (nOmega * nRadius);
    uint iRadius = (gti - iPhi*nOmega*nRadius) / nOmega;
    uint iOmega = gti - iPhi*nOmega*nRadius - iRadius*nOmega;

    ${my_dtype} omegaLocal = omega[iOmega];

    ${my_dtype}3 coordOnScreen = (${my_dtype}3) { radius[iRadius]*cosPhi[iPhi],
                                                  radius[iRadius]*sinPhi[iPhi],
                                                  distanceToScreen };

    ${my_dtype}3 xLocal, uLocal, rVec, nVec, c1, c2;
    ${my_dtype} time, phase, dPhase, sinPhase, cosPhase, rLocal, rInv, gammaInv;

    ${my_dtype} wpdt = ${f_native}sqrt(wp) * dt;
    ${my_dtype} phasePrev = (${my_dtype}) 0.;
    ${my_dtype}3 spectrLocalRe = (${my_dtype}3) {0., 0., 0.};
    ${my_dtype}3 spectrLocalIm = (${my_dtype}3) {0., 0., 0.};

    uint iSnap, it_glob;
    for (iSnap=0; iSnap<nSnaps; iSnap++)
    {
      if (itStart < itSnaps[iSnap]) break;
    }

    for (uint it=0; it<itEnd-1; it++)
    {
      it_glob = itStart + it;

      if (it<nSteps-1)
      {
        time = (${my_dtype})it_glob * dt;
        xLocal = (${my_dtype}3) {x[it], y[it], z[it]};

        rVec = coordOnScreen - xLocal;
        rLocal = ${f_native}sqrt( dot(rVec, rVec) );

        phase = omegaLocal * (time + rLocal) ;
        dPhase = fabs(phase - phasePrev);
        phasePrev = phase;

        if ( dPhase < (${my_dtype})M_PI )
        {
          rInv = (${my_dtype})1. / rLocal;
          nVec = rInv * rVec;

          uLocal = (${my_dtype}3) {ux[it], uy[it], uz[it]};
          gammaInv = ${f_native}rsqrt( (${my_dtype})1. + dot(uLocal, uLocal) );
          uLocal *= gammaInv;

          sinPhase = ${f_native}sin(phase);
          cosPhase = ${f_native}cos(phase);

          c1 = omegaLocal * rInv * (uLocal - nVec);
          c2 = rInv * rInv * nVec;

          spectrLocalRe += -c1*sinPhase + c2*cosPhase;
          spectrLocalIm +=  c1*cosPhase + c2*sinPhase;
        }
      }

      if (it_glob+2 == itSnaps[iSnap])
      {
        spectrum1_re[gti + nTotal*iSnap] += wpdt * spectrLocalRe.s0;
        spectrum1_im[gti + nTotal*iSnap] += wpdt * spectrLocalIm.s0;

        spectrum2_re[gti + nTotal*iSnap] += wpdt * spectrLocalRe.s1;
        spectrum2_im[gti + nTotal*iSnap] += wpdt * spectrLocalIm.s1;

        spectrum3_re[gti + nTotal*iSnap] += wpdt * spectrLocalRe.s2;
        spectrum3_im[gti + nTotal*iSnap] += wpdt * spectrLocalIm.s2;

        iSnap += 1;
      }
    }
  }
}
