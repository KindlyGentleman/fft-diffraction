import pytest
import numpy as np
from diffraction.grids import create_grid
from diffraction.apertures import make_circular_aperture
from diffraction.propagation import propagate_asm, propagate_fresnel
from diffraction.metrics import calculate_energy

def test_unbandlimited_energy_conservation():
    grid = create_grid(nx=256, ny=256, dx=5.0e-6, dy=5.0e-6)
    field_in = make_circular_aperture(grid, radius=1.0e-4)

    e_in = calculate_energy(field_in, grid.dx, grid.dy)

    # For modest distance where field remains within computational window
    u_asm, _, _ = propagate_asm(field_in, wavelength=5.32e-7, z=0.005, dx=grid.dx, dy=grid.dy, padding=2)
    e_asm = calculate_energy(u_asm, grid.dx, grid.dy)

    rel_diff = abs(e_asm - e_in) / e_in
    assert rel_diff < 0.02, f"Energy discrepancy too large: {rel_diff}"
