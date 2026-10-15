import pytest
import numpy as np
from diffraction.grids import create_grid
from diffraction.apertures import make_circular_aperture, make_rectangular_aperture, make_double_slit_aperture

def test_circular_aperture_area():
    grid = create_grid(nx=256, ny=256, dx=2.0e-6, dy=2.0e-6)
    radius = 1.0e-4
    field = make_circular_aperture(grid, radius=radius)
    numerical_area = np.sum(np.abs(field)) * grid.dx * grid.dy
    exact_area = np.pi * (radius**2)
    relative_err = abs(numerical_area - exact_area) / exact_area
    assert relative_err < 0.02

def test_rectangular_aperture_area():
    grid = create_grid(nx=256, ny=256, dx=2.0e-6, dy=2.0e-6)
    w = 2.0e-4
    h = 3.0e-4
    field = make_rectangular_aperture(grid, width=w, height=h)
    numerical_area = np.sum(np.abs(field)) * grid.dx * grid.dy
    exact_area = w * h
    relative_err = abs(numerical_area - exact_area) / exact_area
    assert relative_err < 0.02

def test_invalid_aperture_dimensions():
    grid = create_grid(nx=64, ny=64, dx=1.0, dy=1.0)
    with pytest.raises(ValueError):
        make_circular_aperture(grid, radius=-1.0)
    with pytest.raises(ValueError):
        make_rectangular_aperture(grid, width=0.0, height=1.0)
