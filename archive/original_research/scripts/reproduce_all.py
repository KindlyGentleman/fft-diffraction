import os
import subprocess
import sys
import time

def run_step(step_name: str, cmd: list):
    print("\n" + "=" * 70)
    print(f"STEP: {step_name}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 70)
    t0 = time.perf_counter()
    res = subprocess.run([sys.executable] + cmd, check=True)
    t1 = time.perf_counter()
    print(f"--> Completed {step_name} in {t1 - t0:.2f} s")

def main():
    print("=" * 70)
    print("REPRODUCE ALL: END-TO-END AUTOMATED SCIENTIFIC PIPELINE")
    print("=" * 70)
    total_start = time.perf_counter()

    # 1. Validation
    run_step("1. Scientific Validation Suite", ["scripts/run_validation.py"])

    # 2. Experiments
    run_step("2. Parameter Sweeps & Benchmarking", ["scripts/run_experiments.py", "--mode", "quick"])

    # 3. Figures
    run_step("3. IEEE Figure Generation", ["scripts/generate_figures.py"])

    # 4. Video
    run_step("4. Video Showcase Render (Draft)", ["scripts/generate_video.py", "--mode", "draft"])

    total_time = time.perf_counter() - total_start
    print("\n" + "=" * 70)
    print(f"ALL REPRODUCIBILITY STEPS COMPLETED SUCCESSFULLY IN {total_time:.1f} s")
    print("=" * 70)

if __name__ == "__main__":
    main()
