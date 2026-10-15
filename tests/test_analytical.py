import pytest
import numpy as np
from diffraction.analytical import jinc, airy_first_zero_radius, FIRST_AIRY_ZERO_Q

def test_jinc_limit_zero():
    q = np.array([0.0, 1e-15, -1e-15])
    val = jinc(q)
    assert np.allclose(val, 1.0, atol=1e-10)

def test_airy_first_zero():
    wv = 5.32e-7
    radius = 1.5e-4
    z = 0.5
    r1 = airy_first_zero_radius(wv, z, radius)
    # Check that 2*J1(k*a*r1/z) / (k*a*r1/z) is near 0
    k = 2.0 * np.pi / wv
    q1 = k * radius * r1 / z
    assert np.isclose(q1, FIRST_AIRY_ZERO_Q, atol=1e-10)
    assert abs(jinc(np.array([q1]))[0]) < 1e-6
