import pytest
import numpy as np
from diffraction.grids import create_grid

def test_grid_creation():
    grid = create_grid(nx=256, ny=128, dx=5.0e-6, dy=10.0e-6)
    assert grid.nx == 256
    assert grid.ny == 128
    assert grid.X.shape == (128, 256)
    assert grid.Y.shape == (128, 256)
    assert grid.FX.shape == (128, 256)
    assert grid.FY.shape == (128, 256)

def test_grid_centering():
    grid = create_grid(nx=64, ny=64, dx=1.0, dy=1.0)
    assert grid.x[32] == 0.0
    assert grid.y[32] == 0.0
    assert grid.fx[32] == 0.0
    assert grid.fy[32] == 0.0

def test_invalid_grid_parameters():
    with pytest.raises(ValueError):
        create_grid(nx=-10, ny=64, dx=1.0, dy=1.0)
    with pytest.raises(ValueError):
        create_grid(nx=64, ny=64, dx=0.0, dy=1.0)
