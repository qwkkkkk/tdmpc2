"""High-resolution, render-only video capture for offline evaluation."""

from pathlib import Path

import numpy as np


def render_highres(env, size):
    """Render the underlying simulator without changing the policy observation."""
    width = height = int(size)
    current = env
    errors = []
    seen = set()

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        render = getattr(current, "render", None)
        if callable(render):
            try:
                frame = render(width=width, height=height)
                frame = _as_rgb_uint8(frame)
                if frame.shape[:2] != (height, width):
                    raise RuntimeError(
                        f"renderer returned {frame.shape[:2]}, expected {(height, width)}"
                    )
                return frame
            except (AttributeError, TypeError, RuntimeError, ValueError) as exc:
                errors.append(f"{type(current).__name__}: {exc}")
        current = getattr(current, "env", None)

    detail = "; ".join(errors[-3:]) if errors else "no render method found"
    raise RuntimeError(f"Could not capture a {size}x{size} RGB frame: {detail}")


def _as_rgb_uint8(frame):
    frame = np.asarray(frame)
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim != 3:
        raise ValueError(f"expected HWC image, got shape {frame.shape}")
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.shape[-1] != 3:
        raise ValueError(f"expected RGB/RGBA image, got shape {frame.shape}")
    if frame.dtype != np.uint8:
        if frame.size and float(frame.max()) <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


class EvalVideoRecorder:
    """Stream simulator renders directly to MP4 to keep memory bounded."""

    def __init__(self, path, size=512, fps=16):
        self.path = Path(path)
        self.size = int(size)
        self.fps = int(fps)
        self._writer = None
        self._disabled = False

    def capture(self, env):
        if self._disabled:
            return
        if self._writer is None:
            try:
                import imageio.v2 as imageio

                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._writer = imageio.get_writer(
                    str(self.path),
                    fps=self.fps,
                    codec="libx264",
                    output_params=["-crf", "18", "-pix_fmt", "yuv420p"],
                )
            except Exception as exc:
                print(f"[warn] video writer unavailable for {self.path}: {exc}")
                self._disabled = True
                return
        try:
            self._writer.append_data(render_highres(env, self.size))
        except Exception as exc:
            print(f"[warn] high-resolution video disabled for {self.path}: {exc}")
            self.close()
            self._disabled = True

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None
