import pytest
import numpy as np
from diffraction.grids import create_grid
from diffraction.apertures import make_circular_aperture
from diffraction.propagation import propagate_asm, propagate_blas, propagate_fresnel
from diffraction.metrics import relative_l2_error

def test_zero_distance_reproduction():
    grid = create_grid(nx=128, ny=128, dx=5.0e-6, dy=5.0e-6)
    field_in = make_circular_aperture(grid, radius=1.0e-4)

    for method, fn in [("asm", propagate_asm), ("blas", propagate_blas), ("fresnel", propagate_fresnel)]:
        u_out, _, _ = fn(field_in, wavelength=5.32e-7, z=0.0, dx=grid.dx, dy=grid.dy, padding=2)
        err = relative_l2_error(u_out, field_in)
        assert err < 1e-12, f"Method {method} failed zero distance check with error {err}"

def test_plane_wave_phase_shift():
    grid = create_grid(nx=128, ny=128, dx=5.0e-6, dy=5.0e-6)
    pw = np.ones((128, 128), dtype=np.complex128)
    wavelength = 5.32e-7
    z = 0.02
    k = 2.0 * np.pi / wavelength
    expected = np.exp(1j * k * z) * pw

    for method, fn in [("asm", propagate_asm), ("blas", propagate_blas), ("fresnel", propagate_fresnel)]:
        u_out, _, _ = fn(pw, wavelength=wavelength, z=z, dx=grid.dx, dy=grid.dy, padding=1)
        err = relative_l2_error(u_out, expected)
        assert err < 1e-10, f"Method {method} failed plane wave test with error {err}"

def test_negative_distance_rejection():
    grid = create_grid(nx=64, ny=64, dx=5.0e-6, dy=5.0e-6)
    field_in = np.ones((64, 64), dtype=np.complex128)
    with pytest.raises(ValueError):
        propagate_blas(field_in, wavelength=5.32e-7, z=-0.01, dx=grid.dx, dy=grid.dy)
