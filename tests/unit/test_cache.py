import numpy as np
import sympy as sp

from particlesim.symbolic.cache import compile_cached, key_for


def test_compile_cached_hits_on_second_call(tmp_path, monkeypatch):
    monkeypatch.setenv("PARTICLESIM_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("PARTICLESIM_NO_CACHE", raising=False)
    x, y = sp.symbols("x y")
    calls = []

    def build():
        calls.append(1)
        return [x**2 + y, sp.sin(x) * y]

    key = key_for("test", "x**2+y", 1)
    f1, hit1 = compile_cached(key, [x, y], build)
    f2, hit2 = compile_cached(key, [x, y], build)
    assert (hit1, hit2) == (False, True)
    assert len(calls) == 1
    X = np.linspace(0, 1, 5)
    Y = np.linspace(1, 2, 5)
    np.testing.assert_allclose(f1(X, Y), f2(X, Y))
    np.testing.assert_allclose(f2(X, Y)[0], X**2 + Y)


def test_cache_disabled_by_env(tmp_path, monkeypatch):
    monkeypatch.setenv("PARTICLESIM_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("PARTICLESIM_NO_CACHE", "1")
    x = sp.Symbol("x")
    _, hit = compile_cached(key_for("k"), [x], lambda: [x])
    _, hit2 = compile_cached(key_for("k"), [x], lambda: [x])
    assert not hit and not hit2
    assert not list(tmp_path.iterdir())


def test_key_is_stable_and_sensitive():
    assert key_for("a", 1, (2, 3)) == key_for("a", 1, (2, 3))
    assert key_for("a", 1) != key_for("a", 2)
