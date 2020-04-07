import numpy as np

from synchrad.calc import SynchRad
from synchrad.utils import J_in_um
import sys, time

# Undulator
K0 = 0.1  # Strength
Periods = 50  # Number of periods

# Particles and tracks characteristics
Np = 24  # Number of particles
g0 = 100.0  # Mean Lorentz factor
dg = 1e-4 * g0  # Energy spread
StepsPerPeriod = 32  # Track temporal resolution

# Mean particles longitudinal velocity and central wavenumber
gg = g0 / (1.0 + K0 ** 2 / 2) ** 0.5
vb = (1.0 - gg ** -2) ** 0.5
k_res = 2 * gg ** 2

# Time step and total number of steps
dt = 1.0 / StepsPerPeriod
Steps2Do = int((Periods + 2) / dt) + 1

# Calculating the particles orbits (simplified)
def uxFunctionUndulator(z):
    val = K0 * np.sin(2 * np.pi * z)
    val *= (z > 0) * (z < 1.5) * z / 1.5 + (z > 1.5)
    val *= (z > Periods - 1.5) * (z < Periods) * (Periods - z) / 1.5 + (
        z < Periods - 1.5
    )
    return val


t = np.r_[-1 : Periods + 1 : Steps2Do * 1j]
y = np.zeros_like(t)
uy = np.zeros_like(t)
w = 1.0

particleTracks = []
for g0_p in g0 + dg * np.random.randn(Np):
    gg = g0_p / (1.0 + K0 ** 2 / 2) ** 0.5
    vb = (1.0 - gg ** -2) ** 0.5
    z = vb * t
    ux = uxFunctionUndulator(z - 0.5 * dt)
    uz = (g0 ** 2 - 1 - ux ** 2) ** 0.5
    x = ux[0] / g0_p * dt / 2 + np.cumsum(ux / g0_p) * dt

    particleTracks.append([x, y, z, ux, uy, uz, w, 0])

# Define calculator input

calc_input = {
    "grid": [
        (0.02 * k_res, 1.1 * k_res),  # Wavenumber mapping region
        (0, 2.0 / g0),  # Elevation (theta) angle region
        (0.0, 2 * np.pi),  # Rotation (phi) angle
        (128, 32, 32),
    ],  # Corresponding resolutions
    "timeStep": dt,  # normalized timestep
    # 'ctx':'mpi',                        # OpenCL context (leave commented to be asked)
}

print("Running default mode with double precision")

calc = SynchRad(calc_input)

t0 = time.time()
calc.calculate_spectrum(particleTracks.copy(), comp="total", Np_max=Np)

if calc.rank == 0:
    print(
        "Done {:s}field spectrum from {:d} particle(s) in {:g} sec".format(
            calc.Args["mode"], Np, (time.time() - t0)
        )
    )
    energyModel = calc.get_energy(lambda0_um=1)
    energyTheory = (
        Np
        * k_res
        * J_in_um
        * (7 * np.pi / 24)
        / 137.0
        * K0 ** 2
        * (1 + K0 ** 2 / 2)
        * Periods
    )
    var = abs(energyModel - energyTheory) / energyTheory

    print("Deviation from analytic estimate is {:.2f}%".format(var * 100))


print("\nRunning light mode with single precision and native function support")

calc.Args["dtype"] = "float"
calc.Args["native"] = True

calc._init_args(calc.Args)
calc._init_data()
calc._compile_kernels()

t0 = time.time()
calc.calculate_spectrum(particleTracks.copy(), comp="total", Np_max=Np)

if calc.rank == 0:
    print(
        "Done {:s}field spectrum from {:d} particle(s) in {:g} sec".format(
            calc.Args["mode"], Np, (time.time() - t0)
        )
    )

    energyModel = calc.get_energy(lambda0_um=1)
    energyTheory = (
        Np
        * k_res
        * J_in_um
        * (7 * np.pi / 24)
        / 137.0
        * K0 ** 2
        * (1 + K0 ** 2 / 2)
        * Periods
    )
    var = abs(energyModel - energyTheory) / energyTheory

    print("Deviation from analytic estimate is {:.2f}%".format(var * 100))
