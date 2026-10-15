import json
import os
import sys

from diffraction.validation import run_full_validation_suite
from diffraction.utils import get_system_info

def main():
    print("=" * 70)
    print("EUROPHOTONICS SCALAR DIFFRACTION - SCIENTIFIC VALIDATION SUITE")
    print("=" * 70)

    sys_info = get_system_info()
    print(f"Host OS: {sys_info['os']}")
    print(f"Processor: {sys_info['processor']}")
    print(f"Python: {sys_info['python_version']}  |  NumPy: {sys_info['numpy_version']}\n")

    results = run_full_validation_suite()

    print(f"{'TEST NAME':<38} | {'STATUS':<8} | {'KEY METRIC'}")
    print("-" * 70)

    for test_key, res in results.items():
        if test_key == "ALL_TESTS_PASSED":
            continue
        test_name = res.get("test", test_key)
        passed = res.get("passed", False)
        status_str = "PASS" if passed else "FAIL"

        metric_str = ""
        if "error_blas" in res:
            metric_str = f"BL-ASM err={res['error_blas']:.2e}"
        elif "roundtrip_error" in res:
            metric_str = f"roundtrip err={res['roundtrip_error']:.2e}"
        elif "r1_relative_error" in res:
            metric_str = f"r1 err={res['r1_relative_error']*100:.2f}%, prof err={res['profile_l2_error']*100:.2f}%"
        elif "slice_x_error" in res:
            metric_str = f"slice x err={res['slice_x_error']*100:.2f}%, y err={res['slice_y_error']*100:.2f}%"
        elif "relative_l2_error" in res:
            metric_str = f"RS direct int err={res['relative_l2_error']*100:.2f}%"
        elif "discrepancy_blas" in res:
            metric_str = f"energy disc={res['discrepancy_blas']*100:.2f}%"
        elif "intensity_relative_l2_error" in res:
            metric_str = f"waveprop L2 err={res['intensity_relative_l2_error']*100:.4f}%"

        print(f"{test_name:<38} | {status_str:<8} | {metric_str}")

    print("-" * 70)
    all_ok = results.get("ALL_TESTS_PASSED", False)
    overall_status = "ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED"
    print(f"OVERALL VALIDATION STATUS: {overall_status}\n")

    out_dir = os.path.join("outputs", "metrics")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "validation_results.json")

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Detailed validation metrics saved to: {out_file}")

    if not all_ok:
        sys.exit(1)

if __name__ == "__main__":
    main()
