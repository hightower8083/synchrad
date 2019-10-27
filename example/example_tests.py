import time

import attr
import numpy as np
from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable

from synchrad.calc import SynchRad

# Set the environment variable PYOPENCL_CTX='N', where N is the number of the device you want to compute on.
from synchrad.utils import J_in_um


class Undulator:
    def __init__(
        self,
        strength=0.1,
        number_of_periods=50,
        steps_per_period=32,
        number_of_particles=24,
        mean_lorentz_factor=100.0,
    ):
        self.strength = strength

        self.number_of_periods = number_of_periods
        self.steps_per_period = steps_per_period
        self.time_step = 1.0 / self.steps_per_period
        self.steps_to_do = int((self.number_of_periods + 2) / self.time_step) + 1
        self.t = np.r_[-1 : self.number_of_periods + 1 : self.steps_to_do * 1j]

        self.number_of_particles = number_of_particles
        self.mean_lorentz_factor = mean_lorentz_factor
        self.energy_spread = 1e-4 * self.mean_lorentz_factor
        self.particles = []

        self._gg = self._calc_gg(self.mean_lorentz_factor)
        self.longitudinal_velocity = self._longitudinal_velocity(self._gg)
        self.central_wavenumber = 2 * self._gg ** 2

    def _calc_gg(self, lorentz_factor):
        return lorentz_factor / (1.0 + self.strength ** 2 / 2) ** 0.5

    @staticmethod
    def _longitudinal_velocity(gg):
        return (1.0 - gg ** -2) ** 0.5

    def _add_particle(self, p):
        gg = self._calc_gg(p.lorentz_factor)
        longitudinal_velocity = self._longitudinal_velocity(gg)

        z = longitudinal_velocity * self.t
        ux = self._ux(z - 0.5 * self.time_step)
        uz = (self.mean_lorentz_factor ** 2 - 1 - ux ** 2) ** 0.5
        x = (
            ux[0] / p.lorentz_factor * self.time_step / 2
            + np.cumsum(ux / p.lorentz_factor) * self.time_step
        )
        y = self._y(z)
        uy = self._uy(z)

        p.track = Track(x, y, z, ux, uy, uz, p.weight)
        self.particles.append(p)

    def create_particles(self):
        deviation = self.energy_spread * np.random.randn(self.number_of_particles)
        lorentz_factors = self.mean_lorentz_factor + deviation

        for lf in lorentz_factors:
            p = Particle(lorentz_factor=lf)
            self._add_particle(p)

        return self

    def __getitem__(self, p):
        return self.particles[p]

    # Calculating the particles orbits (simplified)
    def _ux(self, z):
        val = self.strength * np.sin(2 * np.pi * z)
        val *= (z > 0) * (z < 1.5) * z / 1.5 + (z > 1.5)
        val *= (z > self.number_of_periods - 1.5) * (z < self.number_of_periods) * (
            self.number_of_periods - z
        ) / 1.5 + (z < self.number_of_periods - 1.5)
        return val

    @staticmethod
    def _y(z):
        return np.zeros_like(z)

    @staticmethod
    def _uy(z):
        return np.zeros_like(z)

    def energy_analytical(self):
        return (
            self.number_of_particles
            * self.central_wavenumber
            * J_in_um
            * (7 * np.pi / 24)
            / 137.0
            * self.strength ** 2
            * (1 + self.strength ** 2 / 2)
            * self.number_of_periods
        )


@attr.s
class Particle:
    lorentz_factor = attr.ib()
    track = attr.ib(default=None)
    weight = attr.ib(default=1.0)


@attr.s
class Track:
    x = attr.ib()
    y = attr.ib()
    z = attr.ib()
    ux = attr.ib()
    uy = attr.ib()
    uz = attr.ib()
    w = attr.ib()

    def aslist(self):
        return [self.x, self.y, self.z, self.ux, self.uy, self.uz, self.w]


u = Undulator()


@attr.s
class Grid:
    from_k = attr.ib(default=0.02 * u.central_wavenumber)
    to_k = attr.ib(default=1.1 * u.central_wavenumber)
    resolution_k = 512

    from_theta = attr.ib(default=0.0)
    to_theta = attr.ib(default=2.0 / u.mean_lorentz_factor)
    resolution_theta = 256

    from_phi = attr.ib(default=0.0)
    to_phi = attr.ib(default=2 * np.pi)
    resolution_phi = 36

    def to_list(self):
        return [
            (self.from_k, self.to_k),
            (self.from_theta, self.to_theta),
            (self.from_phi, self.to_phi),
            (self.resolution_k, self.resolution_theta, self.resolution_phi),
        ]


@attr.s
class CalcInput:
    grid = attr.ib(default=Grid().to_list())
    timeStep = attr.ib(default=u.time_step)
    dtype = attr.ib(default="float")
    native = attr.ib(default=True)


particleTracks = []
for particle in u.create_particles():
    particleTracks.append(particle.track.aslist())

calc_input = attr.asdict(CalcInput())

print("Running native mode with single precision")

calc = SynchRad(calc_input)

t0 = time.time()
calc.calculate_spectrum(
    particleTracks.copy(), comp="total", Np_max=u.number_of_particles
)

print(
    "Done {:s}field spectrum from {:d} particle(s) in {:g} sec".format(
        calc.Args["mode"], u.number_of_particles, (time.time() - t0)
    )
)

energyModel = calc.get_energy(lambda0_um=1)  # need to calculate spectrum first!
var = abs(energyModel - u.energy_analytical()) / u.energy_analytical()
print("Deviation from analytic estimate is {:.2f}%".format(var * 100))

# Plot the spot observed with a band-filter
kFilter = 0.93 * u.central_wavenumber
kBand = 0.003 * kFilter
k = calc.Args["omega"][:, None, None]
spect_filter = np.exp(-(k - kFilter) ** 2 / kBand ** 2)

spot, extent = calc.get_spot_cartesian(
    bins=(600, 600), lambda0_um=2e4, th_part=0.2, spect_filter=spect_filter
)

fig, ax = pyplot.subplots()
ax.imshow(spot.T, extent=extent * 1e3, cmap=pyplot.cm.nipy_spectral)
ax.set_xlabel("mrad", size=14)
ax.set_ylabel("mrad", size=14)
fig.savefig("spot.png")

fig, ax = pyplot.subplots()
img = ax.imshow(calc.Data["radiation"].mean(-1).T, origin="lower", aspect="auto")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(img, cax=cax)
fig.savefig("radiation.png")

# Results on GeForce GTX 950M, OpenCL 1.2, farfield from 24 particles
# single precision, native functions: 35s, 1.83% error
