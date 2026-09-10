import numpy as np


def level_factors(voxel_size, n_levels):
    """Cumulative (z, y, x) downsample factor per pyramid level.

    X and Y halve every level. Z halves only once the XY sample spacing
    has grown to at least the Z spacing (isotropy), so anisotropic
    stacks are not blurred along Z prematurely.
    """
    vx, vy, vz = voxel_size
    xy = min(vx, vy)  # ponytail: assumes vx == vy, true for our scopes
    fz = fy = fx = 1
    rows = [(1, 1, 1)]
    for _ in range(1, n_levels):
        fy *= 2
        fx *= 2
        if xy * fx >= vz * fz:
            fz *= 2
        rows.append((fz, fy, fx))
    return np.array(rows, dtype="i4")


def step_factors(cumulative):
    """Per-level step: how much level k shrinks relative to level k-1."""
    cumulative = np.asarray(cumulative)
    rows = [(1, 1, 1)]
    for k in range(1, len(cumulative)):
        rows.append(tuple(int(v) for v in cumulative[k] // cumulative[k - 1]))
    return np.array(rows, dtype="i4")


def downsample(arr, step, method):
    """Downsample a (z, y, x) array by a per-axis integer ``step``."""
    sz, sy, sx = step
    if method == "decimate":
        return arr[::sz, ::sy, ::sx]
    if method == "mean":
        z = arr.shape[0] - arr.shape[0] % sz
        y = arr.shape[1] - arr.shape[1] % sy
        x = arr.shape[2] - arr.shape[2] % sx
        blocks = arr[:z, :y, :x].reshape(z // sz, sz, y // sy, sy, x // sx, sx)
        return blocks.mean(axis=(1, 3, 5), dtype=np.float32).astype(arr.dtype)
    raise ValueError(f"unknown downsample method: {method!r}")
