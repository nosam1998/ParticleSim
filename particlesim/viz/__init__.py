"""Visualization: figures, movies and the per-run HTML report (Section 5.7)."""

from particlesim.viz.laser_views import (
    BeamMoments,
    beam_moments,
    energy_spectrum,
    field_snapshot,
    growth_history,
    phase_space,
    spectrum_plot,
)
from particlesim.viz.movies import FFmpegMissing, encode, ffmpeg_path, write_frames
from particlesim.viz.report import render_html_report

__all__ = [
    "BeamMoments",
    "FFmpegMissing",
    "beam_moments",
    "encode",
    "energy_spectrum",
    "ffmpeg_path",
    "field_snapshot",
    "growth_history",
    "phase_space",
    "render_html_report",
    "spectrum_plot",
    "write_frames",
]
