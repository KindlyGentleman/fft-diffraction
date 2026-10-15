from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Grid:
    nx: int
    ny: int
    dx: float
    dy: float
    x: np.ndarray
    y: np.ndarray
    fx: np.ndarray
    fy: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    FX: np.ndarray
    FY: np.ndarray
    Sx: float
    Sy: float
    dfx: float
    dfy: float

def create_grid(nx: int, ny: int, dx: float, dy: float) -> Grid:
    """Create a 2D spatial and spectral coordinate grid.
    
    Coordinates are centered so that index N // 2 corresponds to 0.0,
    matching standard fftshift conventions.
    Array axes follow matrix convention: axis 0 is y, axis 1 is x.
    """
    if nx <= 0 or ny <= 0:
        raise ValueError(f"Grid dimensions must be positive integers, got nx={nx}, ny={ny}")
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError(f"Grid spacings must be positive floats, got dx={dx}, dy={dy}")

    Sx = float(nx * dx)
    Sy = float(ny * dy)
    dfx = 1.0 / Sx
    dfy = 1.0 / Sy

    x = (np.arange(nx, dtype=np.float64) - nx / 2.0) * dx
    y = (np.arange(ny, dtype=np.float64) - ny / 2.0) * dy

    fx = (np.arange(nx, dtype=np.float64) - nx / 2.0) * dfx
    fy = (np.arange(ny, dtype=np.float64) - ny / 2.0) * dfy

    X, Y = np.meshgrid(x, y, indexing="xy")
    FX, FY = np.meshgrid(fx, fy, indexing="xy")

    return Grid(
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        x=x,
        y=y,
        fx=fx,
        fy=fy,
        X=X,
        Y=Y,
        FX=FX,
        FY=FY,
        Sx=Sx,
        Sy=Sy,
        dfx=dfx,
        dfy=dfy,
    )
