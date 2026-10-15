from typing import Optional, Dict, Tuple
import numpy as np

def calculate_intensity(field: np.ndarray) -> np.ndarray:
    """Compute physical optical intensity I = |U|^2."""
    return np.abs(field) ** 2

def calculate_energy(field: np.ndarray, dx: float, dy: float, roi: Optional[np.ndarray] = None) -> float:
    """Compute integrated optical energy sum(|U|^2) * dx * dy."""
    intensity = calculate_intensity(field)
    if roi is not None:
        intensity = intensity[roi]
    return float(np.sum(intensity) * dx * dy)

def relative_l2_error(
    pred: np.ndarray,
    ref: np.ndarray,
    roi: Optional[np.ndarray] = None,
) -> float:
    """Compute relative L2 norm error ||pred - ref||_2 / ||ref||_2."""
    if roi is not None:
        p = pred[roi]
        r = ref[roi]
    else:
        p = pred
        r = ref

    ref_norm = np.linalg.norm(r)
    if ref_norm < 1e-15:
        diff_norm = np.linalg.norm(p - r)
        return float(diff_norm)
    return float(np.linalg.norm(p - r) / ref_norm)

def phase_invariant_complex_error(
    pred: np.ndarray,
    ref: np.ndarray,
    roi: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Compute phase-invariant complex relative L2 error and optimal phase offset."""
    if roi is not None:
        p = pred[roi]
        r = ref[roi]
    else:
        p = pred
        r = ref

    inner_prod = np.sum(p * np.conj(r))
    optimal_phase = float(np.angle(inner_prod))
    aligned_p = p * np.exp(-1j * optimal_phase)

    ref_norm = np.linalg.norm(r)
    if ref_norm < 1e-15:
        err = float(np.linalg.norm(aligned_p - r))
    else:
        err = float(np.linalg.norm(aligned_p - r) / ref_norm)
    return err, optimal_phase

def find_first_radial_minimum(
    r_coords: np.ndarray,
    radial_profile: np.ndarray,
    search_range: Optional[Tuple[float, float]] = None,
) -> float:
    """Find the radial position of the first intensity minimum."""
    if search_range is not None:
        r_min, r_max = search_range
        valid = (r_coords >= r_min) & (r_coords <= r_max)
        r_sub = r_coords[valid]
        p_sub = radial_profile[valid]
    else:
        r_sub = r_coords
        p_sub = radial_profile

    if len(p_sub) < 3:
        return float(r_sub[np.argmin(p_sub)])

    diffs = np.diff(p_sub)
    sign_changes = np.where((diffs[:-1] <= 0) & (diffs[1:] > 0))[0]
    if len(sign_changes) > 0:
        idx = sign_changes[0] + 1
        return float(r_sub[idx])
    return float(r_sub[np.argmin(p_sub)])

def calculate_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    dx: float,
    dy: float,
    roi: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Calculate comprehensive diffraction error metrics."""
    pred_int = calculate_intensity(prediction)
    ref_int = calculate_intensity(reference)

    l2_intensity = relative_l2_error(pred_int, ref_int, roi=roi)
    l2_complex = relative_l2_error(prediction, reference, roi=roi)
    l2_phase_inv, optimal_phase = phase_invariant_complex_error(prediction, reference, roi=roi)

    e_pred = calculate_energy(prediction, dx, dy, roi=roi)
    e_ref = calculate_energy(reference, dx, dy, roi=roi)
    energy_discrepancy = (e_pred - e_ref) / e_ref if e_ref > 1e-15 else 0.0

    return {
        "l2_intensity_error": float(l2_intensity),
        "l2_complex_error": float(l2_complex),
        "l2_phase_invariant_error": float(l2_phase_inv),
        "optimal_global_phase": float(optimal_phase),
        "pred_energy": float(e_pred),
        "ref_energy": float(e_ref),
        "energy_discrepancy": float(energy_discrepancy),
    }
