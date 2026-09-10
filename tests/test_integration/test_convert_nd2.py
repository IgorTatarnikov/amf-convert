import os
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import nd2
import numpy as np
import pytest

from amf_convert.pipeline import convert
from amf_convert.pyramid import downsample, level_factors, step_factors

ND2 = os.environ.get("AMF_TEST_ND2")
pytestmark = pytest.mark.skipif(not ND2, reason="set AMF_TEST_ND2")


def test_convert_produces_bdv_h5_and_xml(tmp_path):
    out = tmp_path / "out.h5"
    result = convert(ND2, str(out), n_levels=3, chunk=64)
    assert Path(result) == out
    assert out.exists()

    with h5py.File(str(out), "r") as f:
        assert "s00/resolutions" in f
        assert "t00000/s00/0/cells" in f
        assert "t00000/s00/2/cells" in f

    xml = out.with_suffix(".xml")
    assert xml.exists()
    root = ET.parse(str(xml)).getroot()
    assert root.tag == "SpimData"
    assert root.find(".//ImageLoader").get("format") == "bdv.hdf5"

    # pyramid levels 1..2 must match an iterative (not cumulative) reference
    with nd2.ND2File(ND2) as nf:
        voxel = tuple(nf.voxel_size())
        ref = np.asarray(nf.to_dask()[0, :, 0, :, :])  # tile0 ch0, level 0
    steps = step_factors(level_factors(voxel, 3))
    with h5py.File(str(out), "r") as f:
        for k in range(1, 3):
            ref = downsample(ref, tuple(steps[k]), "decimate")
            np.testing.assert_array_equal(f[f"t00000/s00/{k}/cells"][:], ref)
