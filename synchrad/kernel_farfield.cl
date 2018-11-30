// kernels of far field calculation (total and single component)

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
    ${my_dtype} nVec[3] = { cosTheta[iTheta],
                       sinTheta[iTheta]*sinPhi[iPhi],
                       sinTheta[iTheta]*cosPhi[iPhi] };


    ${my_dtype} xLocal[3];
    ${my_dtype} uLocal[3];
    ${my_dtype} uNextLocal[3];
    ${my_dtype} aLocal[3];

    ${my_dtype} time, phase, dPhase, sinPhase, cosPhase;
    ${my_dtype} c1, c2, amplitude, gammaInv;

    ${my_dtype} dtInv = 1./dt;
    ${my_dtype} wpdt2 = wp*dt*dt;
    ${my_dtype} phasePrev = 0;
    ${my_dtype} spectrLocalRe[3] = {0., 0., 0.};
    ${my_dtype} spectrLocalIm[3] = {0., 0., 0.};

    for (uint it=0; it<nSteps-1; it++){

      time = it * dt;

      xLocal[0] = x[it];
      xLocal[1] = y[it];
      xLocal[2] = z[it];

      phase = omegaLocal * ( time - xLocal[0]*nVec[0]
                                  - xLocal[1]*nVec[1]
                                  - xLocal[2]*nVec[2] );
      dPhase = fabs(phase - phasePrev);
      phasePrev = phase;

      if ( dPhase < (${my_dtype}) M_PI) {

        uLocal[0] = ux[it];
        uNextLocal[0] = ux[it+1];

        uLocal[1] = uy[it];
        uNextLocal[1] = uy[it+1];

        uLocal[2] = uz[it];
        uNextLocal[2] = uz[it+1];

        gammaInv = rsqrt( 1. + uLocal[0]*uLocal[0]
                             + uLocal[1]*uLocal[1]
                             + uLocal[2]*uLocal[2] );

        for (uint i=0; i<3; i++){
          uLocal[i] = uLocal[i] * gammaInv;
        }

        gammaInv = rsqrt( 1. + uNextLocal[0]*uNextLocal[0]
                             + uNextLocal[1]*uNextLocal[1]
                             + uNextLocal[2]*uNextLocal[2] );

        for (uint i=0; i<3; i++){
          uNextLocal[i] = uNextLocal[i] * gammaInv;
          aLocal[i] = ( uNextLocal[i] - uLocal[i] ) * dtInv;
          uLocal[i] = 0.5 * ( uLocal[i] + uNextLocal[i] );
        }

        c1 = aLocal[0]*nVec[0] + aLocal[1]*nVec[1] + aLocal[2]*nVec[2];
        c2 = 1.0 - uLocal[0]*nVec[0] - uLocal[1]*nVec[1] - uLocal[2]*nVec[2];

        c2 = 1.0/c2;
        c1 = c1*c2*c2;

        sinPhase = sin(phase);
        cosPhase = cos(phase);

        for (uint i=0; i<3; i++){
          amplitude = c1*( nVec[i]-uLocal[i] ) - c2*aLocal[i];
          spectrLocalRe[i] += amplitude * cosPhase;
          spectrLocalIm[i] += amplitude * sinPhase;
        }
      }
    }

    spectrum[gti] +=  wpdt2 * ( spectrLocalRe[0]*spectrLocalRe[0]
                              + spectrLocalIm[0]*spectrLocalIm[0]
                              + spectrLocalRe[1]*spectrLocalRe[1]
                              + spectrLocalIm[1]*spectrLocalIm[1]
                              + spectrLocalRe[2]*spectrLocalRe[2]
                              + spectrLocalIm[2]*spectrLocalIm[2] );
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
    ${my_dtype} nVec[3] = { cosTheta[iTheta],
                       sinTheta[iTheta]*sinPhi[iPhi],
                       sinTheta[iTheta]*cosPhi[iPhi] };

    ${my_dtype} xLocal[3];
    ${my_dtype} uLocal[3];
    ${my_dtype} uNextLocal[3];
    ${my_dtype} aLocal[3];

    ${my_dtype} time, phase, dPhase, sinPhase, cosPhase;
    ${my_dtype} c1, c2, amplitude, gammaInv;

    ${my_dtype} dtInv = 1./dt;
    ${my_dtype} wpdt2 = wp*dt*dt;
    ${my_dtype} phasePrev = 0;
    ${my_dtype} spectrLocalRe = 0.0;
    ${my_dtype} spectrLocalIm = 0.0;

    for (uint it=0; it<nSteps-1; it++){

      time = it * dt;

      xLocal[0] = x[it];
      xLocal[1] = y[it];
      xLocal[2] = z[it];

      phase = omegaLocal * ( time - xLocal[0]*nVec[0]
                                  - xLocal[1]*nVec[1]
                                  - xLocal[2]*nVec[2] );
      dPhase = fabs(phase - phasePrev);
      phasePrev = phase;

      if ( dPhase < (${my_dtype}) M_PI) {

        uLocal[0] = ux[it];
        uNextLocal[0] = ux[it+1];

        uLocal[1] = uy[it];
        uNextLocal[1] = uy[it+1];

        uLocal[2] = uz[it];
        uNextLocal[2] = uz[it+1];

        gammaInv = rsqrt( 1. + uLocal[0]*uLocal[0]
                             + uLocal[1]*uLocal[1]
                             + uLocal[2]*uLocal[2] );

        for (uint i=0; i<3; i++){
          uLocal[i] = uLocal[i] * gammaInv;
        }

        gammaInv = rsqrt( 1. + uNextLocal[0]*uNextLocal[0]
                             + uNextLocal[1]*uNextLocal[1]
                             + uNextLocal[2]*uNextLocal[2] );

        for (uint i=0; i<3; i++){
          uNextLocal[i] = uNextLocal[i] * gammaInv;
          aLocal[i] = ( uNextLocal[i] - uLocal[i] ) * dtInv;
          uLocal[i] = 0.5 * ( uLocal[i] + uNextLocal[i] );
        }

        c1 = aLocal[0]*nVec[0] + aLocal[1]*nVec[1] + aLocal[2]*nVec[2];
        c2 = 1.0 - uLocal[0]*nVec[0] - uLocal[1]*nVec[1] - uLocal[2]*nVec[2];

        c2 = 1.0/c2;
        c1 = c1*c2*c2;

        sinPhase = sin(phase);
        cosPhase = cos(phase);

        amplitude = c1 * ( nVec[iComponent]-uLocal[iComponent] ) - c2*aLocal[iComponent];
        spectrLocalRe += amplitude * cosPhase;
        spectrLocalIm += amplitude * sinPhase;
      }
    }

    spectrum[gti] +=  wpdt2 * ( spectrLocalRe*spectrLocalRe
                              + spectrLocalIm*spectrLocalIm );
   }
}
