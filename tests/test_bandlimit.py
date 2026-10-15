import pytest
import numpy as np
from diffraction.grids import create_grid
from diffraction.bandlimit import compute_bandlimit_frequencies, make_bandlimit_mask

def test_bandlimit_near_and_far_limits():
    wv = 5.32e-7
    Sx = 2.56e-3
    Sy = 2.56e-3

    # At z = 0, limit should equal 1 / wv
    fx_0, fy_0 = compute_bandlimit_frequencies(wv, 0.0, Sx, Sy)
    assert np.isclose(fx_0, 1.0 / wv)
    assert np.isclose(fy_0, 1.0 / wv)

    # At large z, limit should scale approximately as Sx / (2 * wv * z)
    z_large = 2.0
    fx_far, _ = compute_bandlimit_frequencies(wv, z_large, Sx, Sy)
    fx_approx = Sx / (2.0 * wv * z_large)
    rel_diff = abs(fx_far - fx_approx) / fx_approx
    assert rel_diff < 0.01

def test_bandlimit_mask_shape():
    grid = create_grid(nx=128, ny=128, dx=5.0e-6, dy=5.0e-6)
    mask = make_bandlimit_mask(grid, wavelength=5.32e-7, z=0.05)
    assert mask.shape == (128, 128)
    assert mask.dtype == bool
    assert mask[64, 64] == True
