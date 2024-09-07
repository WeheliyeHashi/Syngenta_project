# %%
import argparse
import os
import numpy as np

from syn_tracker.processes.process_file import main_process
from syn_tracker import MODEL_FILES_DIR


def ensure_file_exists(fname):
    fname = os.path.abspath(fname)
    if not os.path.exists(fname):
        raise FileNotFoundError(f"{fname} does not exist.")
    return fname


def validate_rw_path(rw_path):
    if "RawVideos" not in rw_path:
        raise ValueError("The string does not contain 'RawVideos'.")
    return "The string contains 'RawVideos'."


def local_process(rw_path, FPS, mm_2_px):
    print(f"{rw_path}")

    video_dir_root = ensure_file_exists(rw_path)
    validate_rw_path(rw_path)
    main_process(rw_path, MODEL_FILES_DIR, FPS=FPS, mm_2_px=mm_2_px)


def main():
    parser = argparse.ArgumentParser(description="Track and segment wflies in a video.")
    parser.add_argument(
        "--rw_path", type=str, required=True, help="Path to the Rawvideos path"
    )
    parser.add_argument("--FPS", type=int, default=19, help="Frames per second")
    parser.add_argument(
        "--cal_value",
        type=np.float64,
        default=1 / 115,
        help="mm to pixel conversion factor",
    )

    args = parser.parse_args()
    local_process(args.rw_path, args.FPS, args.cal_value)


if __name__ == "__main__":
    main()


# %%
