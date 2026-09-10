import pytest

from amf_convert.pipeline import convert


def test_convert_rejects_zarr(tmp_path):
    with pytest.raises(NotImplementedError):
        convert("in.nd2", str(tmp_path / "out.zarr"))


def test_convert_rejects_unknown_suffix(tmp_path):
    with pytest.raises(ValueError):
        convert("in.nd2", str(tmp_path / "out.tiff"))
