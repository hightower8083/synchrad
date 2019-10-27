import attr
import numpy as np

from synchrad.calc import SynchRad
from synchrad.utils import J_in_um


@attr.s(frozen=True, slots=True)
class Particle:
    lorentz_factor = attr.ib()
    track = attr.ib(default=None)


@attr.s(frozen=True, slots=True)
class Track:
    x = attr.ib()
    y = attr.ib()
    z = attr.ib()
    ux = attr.ib()
    uy = attr.ib()
    uz = attr.ib()
    w = attr.ib(default=1.0)

    def aslist(self):
        return [self.x, self.y, self.z, self.ux, self.uy, self.uz, self.w]


@attr.s(frozen=True, slots=True)
class Resolution:
    k = attr.ib(default=512, kw_only=True)
    theta = attr.ib(default=256, kw_only=True)
    phi = attr.ib(default=36, kw_only=True)


@attr.s(frozen=True, slots=True)
class Grid:
    k_range = attr.ib()
    theta_range = attr.ib()
    phi_range = attr.ib()
    resolution = attr.ib(default=Resolution())

    def aslist(self):
        return [
            self.k_range,
            self.theta_range,
            self.phi_range,
            (self.resolution.k, self.resolution.theta, self.resolution.phi),
        ]


@attr.s(frozen=True, slots=True)
class SynchRadInput:
    grid = attr.ib()
    timeStep = attr.ib()

    dtype = attr.ib(default="float")
    native = attr.ib(default=True)

    def asdict(self):
        return attr.asdict(self)


class Undulator:
    def __init__(
        self,
        strength=0.1,
        number_of_periods=50,
        steps_per_period=32,
        number_of_particles=24,
        mean_lorentz_factor=100.0,
        resolution=Resolution(k=512, theta=256, phi=36),
        dtype="float",
        native=True,
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

        self._gg = self._compute_gg(self.mean_lorentz_factor)
        self.longitudinal_velocity = self._longitudinal_velocity(self._gg)
        self.central_wavenumber = 2 * self._gg ** 2

        self.grid = Grid(
            k_range=tuple(v * self.central_wavenumber for v in [0.02, 1.1]),
            theta_range=tuple(v / self.mean_lorentz_factor for v in [0.0, 2.0]),
            phi_range=tuple([0.0, 2.0 * np.pi]),
            resolution=resolution,
        )
        self.spectrum_input = SynchRadInput(
            grid=self.grid.aslist(), timeStep=self.time_step, dtype=dtype, native=native
        ).asdict()

        self.synch_rad = None

    def _compute_gg(self, lorentz_factor):
        return lorentz_factor / (1.0 + self.strength ** 2 / 2) ** 0.5

    @staticmethod
    def _longitudinal_velocity(gg):
        return (1.0 - gg ** -2) ** 0.5

    def _compute_particle_track(self, lorentz_factor):
        gg = self._compute_gg(lorentz_factor)
        longitudinal_velocity = self._longitudinal_velocity(gg)

        z = longitudinal_velocity * self.t
        ux = self._ux(z - 0.5 * self.time_step)
        uz = (self.mean_lorentz_factor ** 2 - 1 - ux ** 2) ** 0.5
        x = (
            ux[0] / lorentz_factor * self.time_step / 2
            + np.cumsum(ux / lorentz_factor) * self.time_step
        )
        y = self._y(z)
        uy = self._uy(z)

        return Track(x, y, z, ux, uy, uz)

    def create_particles(self):
        deviation = self.energy_spread * np.random.randn(self.number_of_particles)
        lorentz_factors = self.mean_lorentz_factor + deviation

        for lf in lorentz_factors:
            p = Particle(lorentz_factor=lf, track=self._compute_particle_track(lf))
            self.particles.append(p)

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


def undulator_spectrum(
    *,
    number_of_particles=24,
    resolution=Resolution(k=512, theta=256, phi=36),
    dtype="float",
    native=True,
):
    u = Undulator(
        number_of_particles=number_of_particles,
        resolution=resolution,
        dtype=dtype,
        native=native,
    )
    synch_rad = SynchRad(u.spectrum_input)

    particle_tracks = list()
    for particle in u.create_particles():
        particle_tracks.append(particle.track.aslist())

    synch_rad.calculate_spectrum(
        particle_tracks.copy(), comp="total", Np_max=u.number_of_particles
    )

    u.synch_rad = synch_rad

    return u
