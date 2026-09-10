from dataclasses import dataclass

import nd2
import numpy as np


@dataclass
class Nd2Source:
    file: nd2.ND2File
    dask: object
    has_p: bool
    num_tiles: int
    num_channels: int
    shape_zyx: tuple
    voxel_size: tuple
    stage_positions: list


def _extract_stage_positions(nd2_file, num_tiles):
    """Per-tile stage positions in microns, ordered to match the P axis.
    Falls back to zeros when the file has no XY position loop."""
    points = None
    for loop in nd2_file.experiment:
        if getattr(loop, "type", None) == "XYPosLoop":
            points = loop.parameters.points
            break
    if points and len(points) >= num_tiles:
        return [
            (p.stagePositionUm.x, p.stagePositionUm.y, p.stagePositionUm.z)
            for p in points[:num_tiles]
        ]
    return [(0.0, 0.0, 0.0)] * num_tiles


def open_nd2(path):
    f = nd2.ND2File(str(path))
    d = f.to_dask()
    assert d.dtype == np.uint16, f"expected uint16, got {d.dtype}"
    axes = tuple(f.sizes)
    assert axes in (("P", "Z", "C", "Y", "X"), ("Z", "C", "Y", "X")), (
        f"unsupported ND2 axis layout {dict(f.sizes)}; expected (P,)Z,C,Y,X"
    )
    has_p = axes[0] == "P"
    num_tiles = d.shape[0] if has_p else 1
    z, c, y, x = d.shape[-4:]
    return Nd2Source(
        file=f,
        dask=d,
        has_p=has_p,
        num_tiles=num_tiles,
        num_channels=c,
        shape_zyx=(z, y, x),
        voxel_size=tuple(f.voxel_size()),
        stage_positions=_extract_stage_positions(f, num_tiles),
    )


def iter_slabs(src, tile, chunk_z):
    """Yield (z0, slab) with slab shape (z<=chunk_z, C, Y, X), one
    contiguous read each (all channels together — measured ~3x faster
    than per-channel strided reads)."""
    sub = src.dask[tile] if src.has_p else src.dask
    z_total = sub.shape[0]
    for z0 in range(0, z_total, chunk_z):
        z1 = min(z0 + chunk_z, z_total)
        yield z0, np.asarray(sub[z0:z1])
