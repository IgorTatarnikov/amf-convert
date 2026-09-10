import numpy as np
import pytest

from amf_convert.pyramid import downsample, level_factors, step_factors


def test_level_factors_anisotropic():
    # xy = 0.426 um, z = 2.0 um: XY halves every level, Z only catches up at L3
    f = level_factors((0.426, 0.426, 2.0), 4)
    assert f.dtype == np.int32
    np.testing.assert_array_equal(
        f,
        np.array([[1, 1, 1], [1, 2, 2], [1, 4, 4], [2, 8, 8]]),
    )


def test_level_factors_isotropic():
    f = level_factors((1.0, 1.0, 1.0), 3)
    np.testing.assert_array_equal(
        f, np.array([[1, 1, 1], [2, 2, 2], [4, 4, 4]])
    )


def test_step_factors():
    cum = np.array([[1, 1, 1], [1, 2, 2], [2, 4, 4]])
    np.testing.assert_array_equal(
        step_factors(cum),
        np.array([[1, 1, 1], [1, 2, 2], [2, 2, 2]]),
    )


def test_downsample_decimate():
    arr = np.arange(4 * 4 * 4, dtype=np.uint16).reshape(4, 4, 4)
    out = downsample(arr, (1, 2, 2), "decimate")
    assert out.shape == (4, 2, 2)
    np.testing.assert_array_equal(out, arr[::1, ::2, ::2])


def test_downsample_mean():
    arr = np.ones((4, 4, 4), dtype=np.uint16)
    arr[0] = 3  # first z-plane all 3s, rest 1s -> mean over z-pair = 2
    out = downsample(arr, (2, 2, 2), "mean")
    assert out.shape == (2, 2, 2)
    assert out.dtype == np.uint16
    assert out[0, 0, 0] == 2
    assert out[1, 0, 0] == 1


def test_downsample_mean_trims_odd():
    arr = np.ones((5, 5, 5), dtype=np.uint16)
    out = downsample(arr, (2, 2, 2), "mean")
    assert out.shape == (2, 2, 2)
