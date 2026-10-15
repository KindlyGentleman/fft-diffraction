import pytest
import numpy as np
from diffraction.grids import create_grid
from diffraction.apertures import make_circular_aperture
from diffraction.propagation import propagate_blas
from diffraction.analytical import circular_fraunhofer_field
from diffraction.metrics import relative_l2_error

def test_blas_sampling_convergence():
    wavelength = 5.32e-7
    radius = 1.5e-4
    z = 0.500
    fixed_fov = 5.12e-3

    errors = []
    for n in [128, 256, 512]:
        dx = fixed_fov / n
        grid = create_grid(n, n, dx, dx)
        field_in = make_circular_aperture(grid, radius=radius)
        u_blas, _, _ = propagate_blas(field_in, wavelength, z, dx, dx, padding=2)
        u_ana = circular_fraunhofer_field(grid, wavelength, z, radius)
        err = relative_l2_error(u_blas, u_ana)
        errors.append(err)

    # Resolution increase should either reduce or maintain low numerical error
    assert errors[-1] < errors[0]
    assert errors[-1] < 0.25
