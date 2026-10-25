"""Small compiled tree operations, kept separate from the large NR right-hand side.

Unrolling several copies of the symbolic kernel into one compiled Runge--Kutta
step makes XLA compilation impractical. Compiling its inexpensive array
combinations separately removes hundreds of tiny device launches per step
without building that enormous graph.
"""

from functools import cache


def _stage(base, rate, scale):
    return {name: value + scale * rate[name] for name, value in base.items()}


def _finish(base, first, second, third, fourth, step):
    return {
        name: value + (step / 6) * (first[name] + 2 * second[name] + 2 * third[name] + fourth[name])
        for name, value in base.items()
    }


def _hermite(start, start_rate, end, end_rate, theta, step):
    t2, t3 = theta * theta, theta * theta * theta
    h00, h10 = 2 * t3 - 3 * t2 + 1, t3 - 2 * t2 + theta
    h01, h11 = -2 * t3 + 3 * t2, t3 - t2
    return {
        name: h00 * value
        + h10 * step * start_rate[name]
        + h01 * end[name]
        + h11 * step * end_rate[name]
        for name, value in start.items()
    }


@cache
def operations(backend):
    functions = (_stage, _finish, _hermite)
    if backend == "numpy":
        return functions
    if backend != "jax":
        raise ValueError("backend must be 'numpy' or 'jax'")
    import jax

    return tuple(jax.jit(function) for function in functions)
