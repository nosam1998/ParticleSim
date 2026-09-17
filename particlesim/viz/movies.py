"""Field movies through ffmpeg (design doc Section 5.7, Milestone 2).

Frames are handed to ffmpeg on a pipe rather than written to a scratch
directory and globbed back. A run producing ten thousand frames otherwise
leaves ten thousand files behind, and the failure mode of the glob approach
is that a stale frame from a previous run ends up in the middle of a movie,
which nobody notices because the movie plays.

ffmpeg is not a dependency of this package. It is looked up when a movie is
actually asked for, and its absence is reported with the command that would
have run rather than as an opaque file-not-found.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path


class FFmpegMissing(RuntimeError):
    """Raised when no ffmpeg binary can be found."""


def ffmpeg_path(binary: str = "ffmpeg") -> str:
    found = shutil.which(binary)
    if found is None:
        raise FFmpegMissing(
            f"{binary!r} is not on PATH. It is not a dependency of this package, "
            "because a movie is optional and a codec library is not a small thing "
            "to require. Install it from your package manager, or pass the frames "
            "to write_frames() and encode them elsewhere"
        )
    return found


def write_frames(frames: Iterable[bytes], directory: str | Path, stem: str = "frame") -> list[Path]:
    """Write PNG frames to numbered files, for encoding by other means.

    The numbering is zero-padded to the width the frame count needs, so the
    files sort correctly in a shell as well as in Python. Six digits of
    padding on a nine-frame run is not tidiness, it is what stops ``frame10``
    sorting before ``frame9``.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    frames = list(frames)
    width = max(4, len(str(len(frames))))
    out = []
    for index, png in enumerate(frames):
        path = directory / f"{stem}{index:0{width}d}.png"
        path.write_bytes(png)
        out.append(path)
    return out


def encode(
    frames: Iterable[bytes],
    path: str | Path,
    fps: int = 24,
    crf: int = 18,
    binary: str = "ffmpeg",
) -> Path:
    """Encode PNG frames into an H.264 file.

    ``crf`` is the quality knob, lower being better; 18 is visually lossless
    for the flat colours a field plot has. The pixel format is forced to
    ``yuv420p`` because the default for PNG input is one that several players
    and most browsers refuse, and the resulting file looks broken rather than
    reporting anything.
    """
    # Validate the argument before requiring an external tool: asking for
    # ffmpeg to encode nothing should report the nothing, not the missing
    # ffmpeg.
    iterator: Iterator[bytes] = iter(frames)
    first = next(iterator, None)
    if first is None:
        raise ValueError("no frames to encode")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_path(binary),
        "-y",
        "-loglevel",
        "error",
        "-f",
        "image2pipe",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    try:
        process.stdin.write(first)
        for frame in iterator:
            process.stdin.write(frame)
        process.stdin.close()
    except BrokenPipeError:  # pragma: no cover - ffmpeg died early
        pass
    _, errors = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(
            f"ffmpeg exited {process.returncode}: {errors.decode(errors='replace').strip()}"
        )
    return path
