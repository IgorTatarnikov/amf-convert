import os
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import pytest

from amf_convert.pipeline import convert

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
