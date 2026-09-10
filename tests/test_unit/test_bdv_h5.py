import h5py
import numpy as np

from amf_convert.bdv_h5 import BdvH5Writer
from amf_convert.pyramid import downsample, level_factors, step_factors


def _slabs(vol, chunk_z):
    # vol: (Z, C, Y, X)
    for z0 in range(0, vol.shape[0], chunk_z):
        yield z0, vol[z0:z0 + chunk_z]


def test_writer_layout_and_pyramid(tmp_path):
    z, c, y, x = 8, 2, 8, 8
    rng = np.random.default_rng(0)
    vol = rng.integers(0, 4096, size=(z, c, y, x), dtype=np.uint16)
    cum = level_factors((1.0, 1.0, 1.0), 2)  # [[1,1,1],[2,2,2]]

    h5_path = tmp_path / "out.h5"
    w = BdvH5Writer(str(h5_path), 1, c, (z, y, x), cum, chunk=4,
                    downsample="decimate")
    w.write_tile(0, _slabs(vol, 4))
    w.close()

    with h5py.File(str(h5_path), "r") as f:
        # metadata datasets, reversed to (x, y, z)
        np.testing.assert_array_equal(f["s00/resolutions"][:], cum[:, ::-1])
        np.testing.assert_array_equal(f["s01/resolutions"][:], cum[:, ::-1])
        assert f["s00/subdivisions"][:].tolist() == [[4, 4, 4], [4, 4, 4]]

        for j in range(c):
            l0 = f[f"t00000/s0{j}/0/cells"]
            assert l0.shape == (z, y, x)
            assert l0.chunks == (4, 4, 4)
            np.testing.assert_array_equal(l0[:], vol[:, j])

            l1 = f[f"t00000/s0{j}/1/cells"]
            expected = downsample(vol[:, j], tuple(step_factors(cum)[1]),
                                  "decimate")
            assert l1.shape == expected.shape
            np.testing.assert_array_equal(l1[:], expected)


def test_writer_chunks_clamped_to_small_levels(tmp_path):
    z, c, y, x = 4, 1, 4, 4
    vol = np.ones((z, c, y, x), dtype=np.uint16)
    cum = level_factors((1.0, 1.0, 1.0), 2)
    h5_path = tmp_path / "s.h5"
    w = BdvH5Writer(str(h5_path), 1, c, (z, y, x), cum, chunk=4)
    w.write_tile(0, _slabs(vol, 4))
    w.close()
    with h5py.File(str(h5_path), "r") as f:
        l1 = f["t00000/s00/1/cells"]
        assert l1.shape == (2, 2, 2)
        assert l1.chunks == (2, 2, 2)  # clamped from 4
