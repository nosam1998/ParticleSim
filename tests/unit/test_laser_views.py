"""Views, beam diagnostics and movies for particle-in-cell runs (Section 5.7)."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from particlesim.viz import (
    FFmpegMissing,
    beam_moments,
    encode,
    energy_spectrum,
    ffmpeg_path,
    field_snapshot,
    growth_history,
    phase_space,
    spectrum_plot,
    write_frames,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# --- beam moments -----------------------------------------------------------


def test_emittance_of_an_uncorrelated_beam_is_the_product_of_its_spreads():
    """With no correlation, ``eps = sigma_x sigma_p`` exactly. That is the
    definition, so a value that is not this one is not emittance."""
    rng = np.random.default_rng(0)
    n = 200_000
    x = rng.normal(0.0, 2.0, n)
    p = rng.normal(0.0, 0.5, n)
    moments = beam_moments(x, p)
    assert moments.rms_position == pytest.approx(2.0, rel=0.01)
    assert moments.rms_momentum == pytest.approx(0.5, rel=0.01)
    assert moments.emittance == pytest.approx(1.0, rel=0.02)


def test_a_perfectly_chirped_beam_has_zero_emittance():
    """A beam whose momentum is a linear function of position occupies no
    area in phase space, however large it looks in either projection. This
    separates emittance from a beam size, which is the whole reason the
    quantity exists.

    It is also where the determinant cancels: evaluating
    ``<x^2><p^2> - <xp>^2`` directly returns 4e-8 here, eight orders above
    where the answer belongs. Regressing the momentum on the position and
    taking the residual spread gives the same determinant without the
    subtraction.
    """
    x = np.linspace(-3.0, 3.0, 5000)
    moments = beam_moments(x, 0.7 * x)
    assert moments.rms_position > 1.0
    assert moments.rms_momentum > 1.0
    assert moments.emittance < 1e-14


def test_moments_respect_particle_weights():
    x = np.array([-1.0, 1.0, 1.0])
    p = np.array([0.0, 0.0, 0.0])
    heavy = beam_moments(x, p, weight=np.array([1.0, 1.0, 1e6]))
    assert heavy.mean_position == pytest.approx(1.0, rel=1e-5)
    even = beam_moments(x, p, weight=np.ones(3))
    assert even.mean_position == pytest.approx(1.0 / 3.0)


def test_moments_validate_their_inputs():
    with pytest.raises(ValueError, match="same shape"):
        beam_moments(np.zeros(4), np.zeros(3))
    with pytest.raises(ValueError, match="at least two particles"):
        beam_moments(np.zeros(1), np.zeros(1))
    with pytest.raises(ValueError, match="total weight must be positive"):
        beam_moments(np.zeros(3), np.zeros(3), weight=np.zeros(3))


def test_emittance_is_never_negative_for_a_degenerate_beam():
    """The determinant can go slightly negative through round-off on a beam
    that is exactly correlated, and a negative area under a square root is a
    crash rather than a number."""
    x = np.linspace(0.0, 1.0, 1000)
    assert beam_moments(x, x).emittance >= 0.0


# --- spectra ----------------------------------------------------------------


def test_a_monoenergetic_beam_lands_in_one_bin():
    """Everything in a single bin, which is the property. Its centre is not
    the beam's energy: numpy widens a zero-width range by half either side,
    so the value falls on an edge."""
    centres, counts = energy_spectrum(np.full(1000, 0.5), bins=32)
    assert counts.sum() == 1000
    assert np.count_nonzero(counts) == 1
    expected = np.sqrt(1 + 0.25) - 1
    width = centres[1] - centres[0]
    assert abs(centres[int(counts.argmax())] - expected) <= width


def test_the_spectrum_resolves_a_spread_beam():
    momentum = np.linspace(0.0, 2.0, 20_000)
    centres, counts = energy_spectrum(momentum, bins=40)
    assert counts.sum() == 20_000
    assert centres[0] < np.sqrt(1 + 1.0) - 1 < centres[-1]
    # Monotone energy in momentum, so the lowest bin holds the slowest.
    assert counts[0] > 0 and counts[-1] > 0


def test_spectrum_accepts_three_component_momenta():
    momentum = np.zeros((500, 3))
    momentum[:, 0] = 1.0
    centres, counts = energy_spectrum(momentum, bins=8)
    assert counts.sum() == 500
    assert np.count_nonzero(counts) == 1
    width = centres[1] - centres[0]
    assert abs(centres[int(counts.argmax())] - (np.sqrt(2) - 1)) <= width


# --- figures ----------------------------------------------------------------


def test_phase_space_renders_a_png():
    rng = np.random.default_rng(1)
    png = phase_space(rng.normal(size=500), rng.normal(size=500))
    assert png.startswith(PNG_MAGIC)


def test_phase_space_switches_to_a_density_when_scatter_stops_carrying_information():
    """Past the threshold a scatter is a solid block of ink. The switch has
    to happen on its own, because nothing about the resulting picture tells
    the reader it has stopped meaning anything."""
    rng = np.random.default_rng(2)
    n = 5000
    dense = phase_space(rng.normal(size=n), rng.normal(size=n), scatter_limit=100, bins=32)
    sparse = phase_space(rng.normal(size=n), rng.normal(size=n), scatter_limit=n + 1)
    assert dense.startswith(PNG_MAGIC) and sparse.startswith(PNG_MAGIC)
    assert dense != sparse


def test_phase_space_validates_its_inputs():
    with pytest.raises(ValueError, match="same number of entries"):
        phase_space(np.zeros(4), np.zeros(5))


def test_the_other_figures_render():
    x = np.linspace(0, 1, 40)
    assert field_snapshot(x, {"Ex": np.sin(x), "Ez": np.cos(x)}).startswith(PNG_MAGIC)
    assert spectrum_plot(*energy_spectrum(np.linspace(0, 1, 100))).startswith(PNG_MAGIC)
    times = np.linspace(0, 10, 200)
    assert growth_history(times, np.exp(0.3 * times)).startswith(PNG_MAGIC)
    assert growth_history(times, np.exp(0.3 * times), rate=0.3).startswith(PNG_MAGIC)


def test_growth_history_survives_an_amplitude_of_exactly_zero():
    """A log axis and a zero are a crash unless something handles it, and a
    mode amplitude passes through zero whenever the mode does."""
    times = np.linspace(0, 1, 50)
    amplitudes = np.zeros(50)
    assert growth_history(times, amplitudes).startswith(PNG_MAGIC)


# --- movies -----------------------------------------------------------------


def test_missing_ffmpeg_says_what_was_looked_for():
    with pytest.raises(FFmpegMissing, match="not on PATH"):
        ffmpeg_path("definitely-not-a-real-binary-name")


def test_frames_are_numbered_so_they_sort(tmp_path: Path):
    """Six digits on a nine-frame run is not tidiness. Without the padding
    ``frame10`` sorts before ``frame9`` in every shell there is."""
    frames = [PNG_MAGIC + bytes([i]) for i in range(12)]
    written = write_frames(frames, tmp_path)
    assert [p.name for p in written][:2] == ["frame0000.png", "frame0001.png"]
    assert sorted(p.name for p in written) == [p.name for p in written]
    assert written[-1].read_bytes() == frames[-1]


def test_encoding_nothing_is_refused():
    with pytest.raises(ValueError, match="no frames"):
        encode([], "/tmp/does-not-matter.mp4")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_a_movie_is_produced_and_plays_as_h264(tmp_path: Path):
    plt = pytest.importorskip("matplotlib.pyplot")
    frames = []
    for shift in range(8):
        frames.append(
            field_snapshot(np.linspace(0, 1, 30), {"E": np.sin(np.linspace(0, 1, 30) * 6 + shift)})
        )
    del plt
    out = encode(frames, tmp_path / "movie.mp4", fps=8)
    assert out.exists() and out.stat().st_size > 0
