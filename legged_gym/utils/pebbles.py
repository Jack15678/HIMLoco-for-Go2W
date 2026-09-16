"""Fixed half-ellipsoid terrain, shared by training and held-out evaluation."""
import numpy as np


def pebble_heightfield(length, width, horizontal_scale, vertical_scale, height,
                       density, rng, platform_size=3.):
    if min(length, width, horizontal_scale, vertical_scale, height) <= 0 or density < 0:
        raise ValueError('Invalid pebble geometry')
    h = np.zeros((round(length / horizontal_scale) + 1,
                  round(width / horizontal_scale) + 1), dtype=np.float32)
    for stone in range(round(length * width * density)):
        a = rng.uniform(.07, .16)
        b = a if stone % 3 == 0 else rng.uniform(.06, .14)
        radius = max(a, b)
        cx, cy = rng.uniform(radius + .05, length - radius - .05), rng.uniform(radius + .05, width - radius - .05)
        if platform_size and abs(cx - length / 2) < platform_size / 2 + radius and abs(cy - width / 2) < platform_size / 2 + radius:
            continue
        angle, top = rng.uniform(0, 2 * np.pi), rng.uniform(.45, 1.) * height
        i0, i1 = max(0, int((cx-radius)/horizontal_scale)), min(h.shape[0], int((cx+radius)/horizontal_scale)+2)
        j0, j1 = max(0, int((cy-radius)/horizontal_scale)), min(h.shape[1], int((cy+radius)/horizontal_scale)+2)
        x = np.arange(i0, i1)[:, None] * horizontal_scale - cx
        y = np.arange(j0, j1)[None, :] * horizontal_scale - cy
        u, v = x*np.cos(angle)+y*np.sin(angle), -x*np.sin(angle)+y*np.cos(angle)
        cap = top * np.sqrt(np.maximum(0., 1.-(u/a)**2-(v/b)**2))
        h[i0:i1, j0:j1] = np.maximum(h[i0:i1, j0:j1], cap)
    return np.rint(h / vertical_scale).astype(np.int16)
