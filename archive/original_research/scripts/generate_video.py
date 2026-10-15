import argparse
import os
import time

from diffraction.animation import generate_showcase_video

def main():
    parser = argparse.ArgumentParser(description="Generate animated diffraction video showcase")
    parser.add_argument("--mode", choices=["draft", "full"], default="full", help="Render mode")
    parser.add_argument("--output", type=str, default=None, help="Output MP4 path")
    args = parser.parse_args()

    print("=" * 70)
    print(f"GENERATING ANIMATED DIFFRACTION SHOWCASE (Mode: {args.mode.upper()})")
    print("=" * 70)

    if args.mode == "draft":
        out_path = args.output or "outputs/video/diffraction_showcase_draft.mp4"
        duration = 15.0
        grid_n = 128
        dx = 1.0e-5
        fps = 20
    else:
        out_path = args.output or "outputs/video/diffraction_showcase_1080p.mp4"
        duration = 75.0
        grid_n = 256
        dx = 6.0e-6
        fps = 24

    print(f"Target Video Path: {out_path}")
    print(f"Resolution: 1920x1080  |  FPS: {fps}  |  Target Duration: {duration:.1f} s")
    print(f"Simulation Grid: {grid_n}x{grid_n} (dx = {dx*1e6:.1f} um)\n")

    t0 = time.perf_counter()
    res_path = generate_showcase_video(
        output_path=out_path,
        fps=fps,
        total_duration_sec=duration,
        grid_n=grid_n,
        dx=dx,
    )
    t1 = time.perf_counter()

    file_size_mb = os.path.getsize(res_path) / (1024.0 * 1024.0)
    print(f"\nVideo rendered successfully in {t1 - t0:.1f} seconds.")
    print(f"Output File: {res_path} ({file_size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
