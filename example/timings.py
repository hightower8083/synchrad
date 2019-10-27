from timeit import timeit, repeat
from undulator.api import undulator_spectrum
import numpy as np
from matplotlib import pyplot

pyplot.style.use("ggplot")


def time_function(name, arg, n=1):
    f = lambda x: name + "(" + str(x) + ")"
    g = globals()

    time = timeit(f(arg), globals=g, number=n)
    return time


def get_times(name, xs, n=1):
    f = lambda x: name + "(" + str(x) + ")"
    g = globals()

    samples = []
    for _ in range(n):
        times = lambda x: repeat(f(x), globals=g, number=1, repeat=n)
        samples.append([np.median(times(x)) for x in xs])
    ys = [np.median(sample) for sample in zip(*samples)]

    return ys


def plot_times(xs, ys, order=None, pct=0.1, ax=pyplot.gca()):
    ax.plot(xs, ys, marker="o")

    if order:
        slopes = [y / order(x) for (x, y) in zip(xs, ys)]
        for slope in (np.percentile(slopes, pct), np.percentile(slopes, 100 - pct)):
            ax.plot(xs, [slope * order(x) for x in xs], linewidth=3)


myspectrum = lambda nop: undulator_spectrum(number_of_particles=nop)

# t = time_function("myspectrum", 24)
# print(t)

if __name__ == "__main__":
    xs = range(10, 26, 4)
    ys = get_times("myspectrum", xs, n=1)

    fig, ax = pyplot.subplots()
    plot_times(xs, ys, ax=ax)
    ax.set(xlabel="Number of particles", ylabel="Time (s)")
    fig.savefig("scaling.png")
