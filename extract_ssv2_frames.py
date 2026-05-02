#!/usr/bin/env python3
"""
Extract SSv2 frames at 30 FPS following the official TimeSformer instructions.

Official command: ffmpeg -i "${video}" -r 30 -q:v 1 "${out_name}"
Output format:    {video_id}/{video_id}_%06d.jpg

Usage:
  python extract_ssv2_frames.py \
      --src_dir data/downstream/ssv2/20bn-something-something-v2 \
      --dst_dir /mnt/data/ssv2_frames \
      --workers 8
"""

import argparse
import os
import subprocess
from multiprocessing import Pool
from pathlib import Path


def extract_one(args):
    src_path, dst_dir = args
    vid_id = Path(src_path).stem  # e.g. "100000"
    out_dir = os.path.join(dst_dir, vid_id)

    if os.path.isdir(out_dir) and len(os.listdir(out_dir)) > 0:
        return vid_id, "skip"

    os.makedirs(out_dir, exist_ok=True)
    out_pattern = os.path.join(out_dir, f"{vid_id}_%06d.jpg")

    try:
        subprocess.run(
            ["ffmpeg", "-i", src_path, "-r", "30", "-q:v", "1", out_pattern],
            capture_output=True, timeout=60)
        nframes = len(os.listdir(out_dir))
        if nframes == 0:
            return vid_id, "empty"
        return vid_id, "ok"
    except subprocess.TimeoutExpired:
        return vid_id, "timeout"
    except Exception as e:
        return vid_id, f"error: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir",
                        default="data/downstream/ssv2/20bn-something-something-v2")
    parser.add_argument("--dst_dir", default="/mnt/data/ssv2_frames")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    videos = sorted([
        os.path.join(args.src_dir, f)
        for f in os.listdir(args.src_dir)
        if f.endswith(".webm")
    ])
    print(f"Found {len(videos)} .webm videos in {args.src_dir}")
    print(f"Extracting frames at 30 FPS to {args.dst_dir}")

    os.makedirs(args.dst_dir, exist_ok=True)
    tasks = [(v, args.dst_dir) for v in videos]

    ok = skip = fail = 0
    with Pool(args.workers) as pool:
        for i, (vid_id, status) in enumerate(pool.imap_unordered(extract_one, tasks, chunksize=16)):
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
            if (i + 1) % 1000 == 0:
                total = i + 1
                print(f"  [{total}/{len(videos)}] ok={ok} skip={skip} fail={fail}")

    print(f"\nDone: {ok} extracted, {skip} skipped, {fail} failed out of {len(videos)}")


if __name__ == "__main__":
    main()
