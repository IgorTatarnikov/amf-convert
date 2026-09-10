import os

import numpy as np
import pytest

from amf_convert.nd2_source import iter_slabs, open_nd2

ND2 = os.environ.get("AMF_TEST_ND2")
pytestmark = pytest.mark.skipif(not ND2, reason="set AMF_TEST_ND2")


def test_open_nd2_metadata():
    src = open_nd2(ND2)
    assert src.num_channels >= 1
    assert src.num_tiles >= 1
    assert len(src.shape_zyx) == 3
    assert len(src.voxel_size) == 3
    assert len(src.stage_positions) == src.num_tiles
    src.file.close()


def test_iter_slabs_shape_and_coverage():
    src = open_nd2(ND2)
    z_total = src.shape_zyx[0]
    seen_z = 0
    for z0, slab in iter_slabs(src, tile=0, chunk_z=64):
        assert slab.ndim == 4  # (z, C, Y, X)
        assert slab.shape[1] == src.num_channels
        assert slab.shape[2:] == src.shape_zyx[1:]
        assert slab.dtype == np.uint16
        assert z0 == seen_z
        seen_z += slab.shape[0]
    assert seen_z == z_total
    src.file.close()
