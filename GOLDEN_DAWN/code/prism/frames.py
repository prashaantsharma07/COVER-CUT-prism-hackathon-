import subprocess

import numpy as np

from .video_io import FFMPEG


def stream_frames(path, fps, size=224):
    """Yields RGB uint8 frames [size, size, 3] at `fps`: short side resized to `size`, centre crop."""
    vf = f"fps={fps},scale={size}:{size}:force_original_aspect_ratio=increase:flags=bicubic,crop={size}:{size}"
    proc = subprocess.Popen([FFMPEG, "-nostdin", "-threads", "0", "-loglevel", "error", "-i", path, "-vf", vf,
                             "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                            stdout=subprocess.PIPE, bufsize=size * size * 3 * 8)
    n = size * size * 3
    try:
        while True:
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(size, size, 3)
    finally:
        proc.stdout.close()
        proc.wait()
