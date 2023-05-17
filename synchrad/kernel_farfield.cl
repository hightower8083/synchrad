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
                  uint itStart,
                  uint itEnd,
                  uint nSteps,
  __global ${my_dtype} *omega,
  __global ${my_dtype} *sinTheta,
  __global ${my_dtype} *cosTheta,
  __global ${my_dtype} *sinPhi,
  __global ${my_dtype} *cosPhi,
                  uint nOmega,
                  uint nTheta,
                  uint nPhi,
           ${my_dtype} dt,
                  uint nSnaps,
  __global        uint *itSnaps)
{
  uint gti = (uint) get_global_id(0);
  uint nTotal = nTheta*nPhi*nOmega;

  if (gti < nTotal)
  {
    uint iPhi = gti / (nOmega * nTheta);
    uint iTheta = (gti - iPhi*nOmega*nTheta) / nOmega;
    uint iOmega = gti - iPhi*nOmega*nTheta - iTheta*nOmega;

    ${my_dtype} omegaLocal = omega[iOmega];
    ${my_dtype}3 nVec = (${my_dtype}3) { sinTheta[iTheta]*cosPhi[iPhi],
                                         sinTheta[iTheta]*sinPhi[iPhi],
                                         cosTheta[iTheta] };

    ${my_dtype}3 xLocal, uLocal, uNextLocal, aLocal, amplitude;
    ${my_dtype} time, phase, dPhase, sinPhase, cosPhase, c1, c2, gammaInv;

    ${my_dtype} dtInv = (${my_dtype})1. / dt;
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

        phase = omegaLocal * (time - dot(xLocal, nVec)) ;
        dPhase = fabs(phase - phasePrev);
        phasePrev = phase;

        if (dPhase < (${my_dtype})M_PI)
        {
          uLocal = (${my_dtype}3) {ux[it], uy[it], uz[it]};
          uNextLocal = (${my_dtype}3) {ux[it+1], uy[it+1], uz[it+1]};

          gammaInv = ${f_native}rsqrt( (${my_dtype})1. + dot(uLocal, uLocal) );
          uLocal *= gammaInv;
          gammaInv = ${f_native}rsqrt( (${my_dtype})1. + dot(uNextLocal, uNextLocal) );
          uNextLocal *= gammaInv;

          aLocal = (uNextLocal - uLocal) * dtInv;
          uLocal = (${my_dtype})0.5 * (uNextLocal + uLocal);

          c1 = dot(aLocal, nVec);
          c2 = (${my_dtype})1. - dot(uLocal, nVec);

          c2 =  (${my_dtype})1. / c2;
          c1 = c1*c2*c2;

          sinPhase = ${f_native}sin(phase);
          cosPhase = ${f_native}cos(phase);

          amplitude = c1*(nVec - uLocal) - c2*aLocal;
          spectrLocalRe += amplitude * cosPhase;
          spectrLocalIm += amplitude * sinPhase;
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
  __global ${my_dtype} *sinTheta,
  __global ${my_dtype} *cosTheta,
  __global ${my_dtype} *sinPhi,
  __global ${my_dtype} *cosPhi,
                  uint nOmega,
                  uint nTheta,
                  uint nPhi,
           ${my_dtype} dt,
                  uint nSnaps,
  __global        uint *itSnaps)
{
  uint gti = (uint) get_global_id(0);
  uint nTotal = nTheta*nPhi*nOmega;

  if (gti < nTotal){

    uint iPhi = gti / (nOmega * nTheta);
    uint iTheta = (gti - iPhi*nOmega*nTheta) / nOmega;
    uint iOmega = gti - iPhi*nOmega*nTheta - iTheta*nOmega;

    ${my_dtype} omegaLocal = omega[iOmega];
    ${my_dtype}3 nVec = (${my_dtype}3) { sinTheta[iTheta]*cosPhi[iPhi],
                                         sinTheta[iTheta]*sinPhi[iPhi],
                                         cosTheta[iTheta] };

    ${my_dtype}3 xLocal, uLocal, uNextLocal, aLocal, amplitude;
    ${my_dtype} time, phase, dPhase, sinPhase, cosPhase, c1, c2, gammaInv;

    ${my_dtype} dtInv = (${my_dtype})1. / dt;
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

        phase = omegaLocal * (time - dot(xLocal, nVec)) ;
        dPhase = fabs(phase - phasePrev);
        phasePrev = phase;

        if (dPhase < (${my_dtype})M_PI)
        {
          uLocal = (${my_dtype}3) {ux[it], uy[it], uz[it]};
          uNextLocal = (${my_dtype}3) {ux[it+1], uy[it+1], uz[it+1]};

          gammaInv = ${f_native}rsqrt( (${my_dtype})1. + dot(uLocal, uLocal) );
          uLocal *= gammaInv;
          gammaInv = ${f_native}rsqrt( (${my_dtype})1. + dot(uNextLocal, uNextLocal) );
          uNextLocal *= gammaInv;

          aLocal = (uNextLocal - uLocal) * dtInv;
          uLocal = (${my_dtype})0.5 * (uNextLocal + uLocal);

          c1 = dot(aLocal, nVec);
          c2 = (${my_dtype})1. - dot(uLocal, nVec);

          c2 =  (${my_dtype})1. / c2;
          c1 = c1*c2*c2;

          sinPhase = ${f_native}sin(phase);
          cosPhase = ${f_native}cos(phase);

          amplitude = c1*(nVec - uLocal) - c2*aLocal;

          spectrLocalRe += amplitude * cosPhase;
          spectrLocalIm += amplitude * sinPhase;
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
  __global ${my_dtype} *sinTheta,
  __global ${my_dtype} *cosTheta,
  __global ${my_dtype} *sinPhi,
  __global ${my_dtype} *cosPhi,
                  uint nOmega,
                  uint nTheta,
                  uint nPhi,
           ${my_dtype} dt,
                  uint nSnaps,
  __global        uint *itSnaps,
  __global ${my_dtype} *FormFactor)
{
  uint gti = (uint) get_global_id(0);
  uint nTotal = nTheta*nPhi*nOmega;

  if (gti < nTotal){

    uint iPhi = gti / (nOmega * nTheta);
    uint iTheta = (gti - iPhi*nOmega*nTheta) / nOmega;
    uint iOmega = gti - iPhi*nOmega*nTheta - iTheta*nOmega;

    ${my_dtype} omegaLocal = omega[iOmega];
    ${my_dtype}3 nVec = (${my_dtype}3) { sinTheta[iTheta]*cosPhi[iPhi],
                                         sinTheta[iTheta]*sinPhi[iPhi],
                                         cosTheta[iTheta] };

    ${my_dtype} FormFactorLocal = FormFactor[iOmega];
    ${my_dtype}3 xLocal, uLocal, uNextLocal, aLocal, amplitude;
    ${my_dtype} time, phase, dPhase, sinPhase, cosPhase, c1, c2, gammaInv;

    ${my_dtype} dtInv = (${my_dtype})1. / dt;
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

        phase = omegaLocal * (time - dot(xLocal, nVec)) ;
        dPhase = fabs(phase - phasePrev);
        phasePrev = phase;

        if (dPhase < (${my_dtype})M_PI)
        {
          uLocal = (${my_dtype}3) {ux[it], uy[it], uz[it]};
          uNextLocal = (${my_dtype}3) {ux[it+1], uy[it+1], uz[it+1]};

          gammaInv = ${f_native}rsqrt( (${my_dtype})1. + dot(uLocal, uLocal) );
          uLocal *= gammaInv;
          gammaInv = ${f_native}rsqrt( (${my_dtype})1. + dot(uNextLocal, uNextLocal) );
          uNextLocal *= gammaInv;

          aLocal = (uNextLocal - uLocal) * dtInv;
          uLocal = (${my_dtype})0.5 * (uNextLocal + uLocal);

          c1 = dot(aLocal, nVec);
          c2 = (${my_dtype})1. - dot(uLocal, nVec);

          c2 =  (${my_dtype})1. / c2;
          c1 = c1*c2*c2;

          sinPhase = ${f_native}sin(phase);
          cosPhase = ${f_native}cos(phase);

          amplitude = c1*(nVec - uLocal) - c2*aLocal;

          spectrLocalRe += amplitude * cosPhase * FormFactorLocal;
          spectrLocalIm += amplitude * sinPhase * FormFactorLocal;
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

__kernel void spheric_comps(
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
  __global ${my_dtype} *sinTheta,
  __global ${my_dtype} *cosTheta,
  __global ${my_dtype} *sinPhi,
  __global ${my_dtype} *cosPhi,
                  uint nOmega,
                  uint nTheta,
                  uint nPhi,
           ${my_dtype} dt,
                  uint nSnaps,
  __global        uint *itSnaps)
{
  uint gti = (uint) get_global_id(0);
  uint nTotal = nTheta*nPhi*nOmega;

  if (gti < nTotal)
  {
    uint iPhi = gti / (nOmega * nTheta);
    uint iTheta = (gti - iPhi*nOmega*nTheta) / nOmega;
    uint iOmega = gti - iPhi*nOmega*nTheta - iTheta*nOmega;

    ${my_dtype} omegaLocal = omega[iOmega];
    ${my_dtype}3 nVec = (${my_dtype}3) { sinTheta[iTheta]*cosPhi[iPhi],
                                         sinTheta[iTheta]*sinPhi[iPhi],
                                         cosTheta[iTheta] };
    ${my_dtype}3 thVec = (${my_dtype}3) { cosTheta[iTheta]*cosPhi[iPhi],
                                          cosTheta[iTheta]*sinPhi[iPhi],
                                         -sinTheta[iTheta] };
    ${my_dtype}3 phVec = (${my_dtype}3) { -sinPhi[iPhi], cosPhi[iPhi], 0.0};

    ${my_dtype}3 xLocal, uLocal, uNextLocal, aLocal, amplitude, amplSpheric;
    ${my_dtype} time, phase, dPhase, sinPhase, cosPhase, c1, c2, gammaInv;

    ${my_dtype} dtInv = (${my_dtype})1. / dt;
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

        phase = omegaLocal * (time - dot(xLocal, nVec)) ;
        dPhase = fabs(phase - phasePrev);
        phasePrev = phase;

        if (dPhase < (${my_dtype})M_PI)
        {
          uLocal = (${my_dtype}3) {ux[it], uy[it], uz[it]};
          uNextLocal = (${my_dtype}3) {ux[it+1], uy[it+1], uz[it+1]};

          gammaInv = ${f_native}rsqrt( (${my_dtype})1. + dot(uLocal, uLocal) );
          uLocal *= gammaInv;
          gammaInv = ${f_native}rsqrt( (${my_dtype})1. + dot(uNextLocal, uNextLocal) );
          uNextLocal *= gammaInv;

          aLocal = (uNextLocal - uLocal) * dtInv;
          uLocal = (${my_dtype})0.5 * (uNextLocal + uLocal);

          c1 = dot(aLocal, nVec);
          c2 = (${my_dtype})1. - dot(uLocal, nVec);

          c2 =  (${my_dtype})1. / c2;
          c1 = c1*c2*c2;

          sinPhase = ${f_native}sin(phase);
          cosPhase = ${f_native}cos(phase);

          amplitude = c1*(nVec - uLocal) - c2*aLocal;

          amplSpheric.s0 = dot(nVec, amplitude);
          amplSpheric.s1 = dot(thVec, amplitude);
          amplSpheric.s2 = dot(phVec, amplitude);
          amplitude = amplSpheric;

          spectrLocalRe += amplitude * cosPhase;
          spectrLocalIm += amplitude * sinPhase;
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

__kernel void spheric_comps_complex(
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
  __global ${my_dtype} *sinTheta,
  __global ${my_dtype} *cosTheta,
  __global ${my_dtype} *sinPhi,
  __global ${my_dtype} *cosPhi,
                  uint nOmega,
                  uint nTheta,
                  uint nPhi,
           ${my_dtype} dt,
                  uint nSnaps,
  __global        uint *itSnaps,
  __global ${my_dtype} *FormFactor)
{
  uint gti = (uint) get_global_id(0);
  uint nTotal = nTheta*nPhi*nOmega;

  if (gti < nTotal)
  {
    uint iPhi = gti / (nOmega * nTheta);
    uint iTheta = (gti - iPhi*nOmega*nTheta) / nOmega;
    uint iOmega = gti - iPhi*nOmega*nTheta - iTheta*nOmega;

    ${my_dtype} omegaLocal = omega[iOmega];
    ${my_dtype}3 nVec = (${my_dtype}3) { sinTheta[iTheta]*cosPhi[iPhi],
                                         sinTheta[iTheta]*sinPhi[iPhi],
                                         cosTheta[iTheta] };
    ${my_dtype}3 thVec = (${my_dtype}3) { cosTheta[iTheta]*cosPhi[iPhi],
                                          cosTheta[iTheta]*sinPhi[iPhi],
                                         -sinTheta[iTheta] };
    ${my_dtype}3 phVec = (${my_dtype}3) { -sinPhi[iPhi], cosPhi[iPhi], 0.0};

    ${my_dtype}3 xLocal, uLocal, uNextLocal, aLocal, amplitude, amplSpheric;
    ${my_dtype} time, phase, dPhase, sinPhase, cosPhase, c1, c2, gammaInv;

    ${my_dtype} dtInv = (${my_dtype})1. / dt;
    ${my_dtype} wpdt =  ${f_native}sqrt(wp) * dt;
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

        phase = omegaLocal * (time - dot(xLocal, nVec)) ;
        dPhase = fabs(phase - phasePrev);
        phasePrev = phase;

        if (dPhase < (${my_dtype})M_PI)
        {
          uLocal = (${my_dtype}3) {ux[it], uy[it], uz[it]};
          uNextLocal = (${my_dtype}3) {ux[it+1], uy[it+1], uz[it+1]};

          gammaInv = ${f_native}rsqrt( (${my_dtype})1. + dot(uLocal, uLocal) );
          uLocal *= gammaInv;
          gammaInv = ${f_native}rsqrt( (${my_dtype})1. + dot(uNextLocal, uNextLocal) );
          uNextLocal *= gammaInv;

          aLocal = (uNextLocal - uLocal) * dtInv;
          uLocal = (${my_dtype})0.5 * (uNextLocal + uLocal);

          c1 = dot(aLocal, nVec);
          c2 = (${my_dtype})1. - dot(uLocal, nVec);

          c2 =  (${my_dtype})1. / c2;
          c1 = c1*c2*c2;

          sinPhase = ${f_native}sin(phase);
          cosPhase = ${f_native}cos(phase);

          amplitude = c1*(nVec - uLocal) - c2*aLocal;

          amplSpheric.s0 = dot(nVec, amplitude);
          amplSpheric.s1 = dot(thVec, amplitude);
          amplSpheric.s2 = dot(phVec, amplitude);
          amplitude = amplSpheric;

          spectrLocalRe += amplitude * cosPhase;
          spectrLocalIm += amplitude * sinPhase;
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
