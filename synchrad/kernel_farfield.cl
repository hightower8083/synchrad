// this is a kernel of far field calculation

__kernel void total(
  __global double *spectrum,
  __global double *x,
  __global double *y,
  __global double *z,
  __global double *ux,
  __global double *uy,
  __global double *uz,
           double wp,
             uint nSteps,
  __global double *omega,
  __global double *cosTheta,
  __global double *sinTheta,
  __global double *cosPhi,
  __global double *sinPhi,
             uint nOmega,
             uint nTheta,
             uint nPhi,
           double dt )
{
  uint gti = (uint) get_global_id(0);

  if (gti < nTheta*nPhi*nOmega)
   {

    uint iPhi = gti / (nOmega * nTheta);
    uint iTheta = (gti - iPhi*nOmega*nTheta) / nOmega;
    uint iOmega = gti - iPhi*nOmega*nTheta - iTheta*nOmega;

    double omegaLocal = omega[iOmega];
    double nVec[3] = { cosTheta[iTheta],
                       sinTheta[iTheta]*sinPhi[iPhi],
                       sinTheta[iTheta]*cosPhi[iPhi] };


    double xLocal[3];
    double uLocal[3];
    double uNextLocal[3];
    double aLocal[3];

    double time, phase, dPhase, sinPhase, cosPhase;
    double c1, c2, amplitude, gammaInv;

    double dtInv = 1./dt;
    double wpdt2 = wp*dt*dt;
    double phasePrev = 0;
    double spectrLocalRe[3] = {0., 0., 0.};
    double spectrLocalIm[3] = {0., 0., 0.};

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

      if ( dPhase < (double) M_PI) {

        uLocal[0] = ux[it];
        uNextLocal[0] = ux[it+1];

        uLocal[1] = uy[it];
        uNextLocal[1] = uy[it+1];

        uLocal[2] = uz[it];
        uNextLocal[2] = uz[it+1];

        gammaInv = 1. / sqrt( 1. + uLocal[0]*uLocal[0]
                                 + uLocal[1]*uLocal[1]
                                 + uLocal[2]*uLocal[2] );

        for (uint i=0; i<3; i++){
          uLocal[i] = uLocal[i] * gammaInv;
        }

        gammaInv = 1. / sqrt( 1. + uNextLocal[0]*uNextLocal[0]
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
  __global double *spectrum,
             uint iComponent,
  __global double *x,
  __global double *y,
  __global double *z,
  __global double *ux,
  __global double *uy,
  __global double *uz,
           double wp,
             uint nSteps,
  __global double *omega,
  __global double *cosTheta,
  __global double *sinTheta,
  __global double *cosPhi,
  __global double *sinPhi,
             uint nOmega,
             uint nTheta,
             uint nPhi,
           double dt )
{
  uint gti = (uint) get_global_id(0);

  if (gti < nTheta*nPhi*nOmega)
   {

    uint iPhi = gti / (nOmega * nTheta);
    uint iTheta = (gti - iPhi*nOmega*nTheta) / nOmega;
    uint iOmega = gti - iPhi*nOmega*nTheta - iTheta*nOmega;

    double omegaLocal = omega[iOmega];
    double nVec[3] = { cosTheta[iTheta],
                       sinTheta[iTheta]*sinPhi[iPhi],
                       sinTheta[iTheta]*cosPhi[iPhi] };

    double xLocal[3];
    double uLocal[3];
    double uNextLocal[3];
    double aLocal[3];

    double time, phase, dPhase, sinPhase, cosPhase;
    double c1, c2, amplitude, gammaInv;

    double dtInv = 1./dt;
    double wpdt2 = wp*dt*dt;
    double phasePrev = 0;
    double spectrLocalRe = 0.0;
    double spectrLocalIm = 0.0;

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

      if ( dPhase < (double) M_PI) {

        uLocal[0] = ux[it];
        uNextLocal[0] = ux[it+1];

        uLocal[1] = uy[it];
        uNextLocal[1] = uy[it+1];

        uLocal[2] = uz[it];
        uNextLocal[2] = uz[it+1];

        gammaInv = 1. / sqrt( 1. + uLocal[0]*uLocal[0]
                                 + uLocal[1]*uLocal[1]
                                 + uLocal[2]*uLocal[2] );

        for (uint i=0; i<3; i++){
          uLocal[i] = uLocal[i] * gammaInv;
        }

        gammaInv = 1. / sqrt( 1. + uNextLocal[0]*uNextLocal[0]
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

        amplitude = c1*( nVec[iComponent]-uLocal[iComponent] ) - c2*aLocal[iComponent];
        spectrLocalRe += amplitude * cosPhase;
        spectrLocalIm += amplitude * sinPhase;
      }
    }

    spectrum[gti] +=  wpdt2 * ( spectrLocalRe*spectrLocalRe
                              + spectrLocalIm*spectrLocalIm );
   }
}
