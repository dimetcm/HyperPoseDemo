#!/usr/bin/env python3
"""Batch-convert FBX animations to ozz runtime archives.

Pipeline per file:
  fbx2ozz (raw, unoptimized, fixed sampling rate) -> ozz_fixup (scale, optimize, build)

One shared skeleton is built from the first input file (or --skeleton-from); every
animation is imported against it, so all inputs must share the same joint names.

Examples:
    uv run Tools/convert_fbx.py Assets/HuggingFace/gbionics__cmu-fbx/animations --out Assets/ozz/cmu
    uv run Tools/convert_fbx.py some/clip.fbx --out Assets/ozz/test --scale 1 --rate 60

Defaults (--scale 0.01, --rate 30) fit the CMU FBX dataset: Blender exports with a
x100 armature scale and a 0 Hz scene frame rate.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FBX2OZZ = ROOT / "build/_deps/ozz-build/src/animation/offline/fbx/fbx2ozz.exe"
DEFAULT_FIXUP = ROOT / "build/Release/ozz_fixup.exe"


def parse_args():
    p = argparse.ArgumentParser(description="Convert FBX animations to ozz archives.")
    p.add_argument("inputs", nargs="+", type=Path, help="FBX files or folders (searched recursively)")
    p.add_argument("--out", type=Path, required=True, help="Output folder")
    p.add_argument("--skeleton-from", type=Path, help="FBX to build the skeleton from (default: first input)")
    p.add_argument("--scale", type=float, default=0.01, help="Global scale (default: 0.01)")
    p.add_argument("--rate", type=float, default=30, help="Sampling rate in Hz (default: 30)")
    p.add_argument("--tolerance", type=float, default=0.001, help="Optimizer tolerance in meters")
    p.add_argument("--jobs", type=int, default=os.cpu_count(), help="Parallel conversions")
    p.add_argument("--limit", type=int, help="Convert only the first N files")
    p.add_argument("--force", action="store_true", help="Reconvert files that are up to date")
    p.add_argument("--fbx2ozz", type=Path, default=DEFAULT_FBX2OZZ, help="Path to fbx2ozz.exe")
    p.add_argument("--fixup", type=Path, default=DEFAULT_FIXUP, help="Path to ozz_fixup.exe")
    return p.parse_args()


def collect(inputs):
    files = []
    for path in inputs:
        if path.is_dir():
            files += sorted(path.rglob("*.fbx"))
        elif path.suffix.lower() == ".fbx":
            files.append(path)
        else:
            sys.exit(f"Not an FBX file or folder: {path}")
    return files


def run(cmd, cwd):
    result = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        log = (result.stdout + result.stderr).strip().splitlines()
        raise RuntimeError("\n    ".join(log[-6:]) or f"exit code {result.returncode}")


def fbx2ozz_config(skeleton, animations=None):
    return json.dumps({
        "skeleton": {"filename": str(skeleton),
                     "import": {"enable": animations is None, "raw": True}},
        "animations": animations or [],
    })


def build_skeleton(args, fbx, out, tools):
    raw = out / ".raw_skeleton.ozz"
    run([tools[0], f"--file={fbx}", f"--config={fbx2ozz_config(raw)}"], out)
    run([tools[1], f"--skeleton={raw}", f"--scale={args.scale}",
         f"--skeleton_output={out / 'skeleton.ozz'}"], out)
    return raw


def convert(args, fbx, out, raw_skeleton, tools):
    target = out / f"{fbx.stem}.ozz"
    if not args.force and target.exists() and target.stat().st_mtime >= fbx.stat().st_mtime:
        return "skipped"
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        animation = {"clip": "*", "filename": "*.ozz", "raw": True,
                     "optimize": False, "sampling_rate": args.rate}
        run([tools[0], f"--file={fbx}", f"--config={fbx2ozz_config(raw_skeleton, [animation])}"], tmp)
        clips = sorted(Path(tmp).glob("*.ozz"))
        if not clips:
            raise RuntimeError("no animation found")
        for i, clip in enumerate(clips):
            # Single-clip files keep the FBX name; extra clips get a numeric suffix.
            dst = target if i == 0 else out / f"{fbx.stem}_{i}.ozz"
            run([tools[1], f"--skeleton={raw_skeleton}", f"--scale={args.scale}",
                 f"--animation={clip}", f"--animation_output={dst}",
                 f"--tolerance={args.tolerance}"], tmp)
    return "converted"


def main():
    args = parse_args()
    tools = (args.fbx2ozz.resolve(), args.fixup.resolve())
    for tool in tools:
        if not tool.exists():
            sys.exit(f"{tool.name} not found at {tool}; build the project first.")

    files = [f.resolve() for f in collect(args.inputs)][: args.limit]
    if not files:
        sys.exit("No FBX files found.")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    skeleton_fbx = (args.skeleton_from or files[0]).resolve()
    print(f"Skeleton from {skeleton_fbx.name}")
    raw_skeleton = build_skeleton(args, skeleton_fbx, out, tools)

    counts = {"converted": 0, "skipped": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(convert, args, f, out, raw_skeleton, tools): f for f in files}
        for done, future in enumerate(as_completed(futures), 1):
            try:
                counts[future.result()] += 1
            except Exception as e:
                counts["failed"] += 1
                print(f"\nFAILED {futures[future].name}:\n    {e}")
            print(f"\r[{done}/{len(files)}] " + ", ".join(f"{k} {v}" for k, v in counts.items()),
                  end="", flush=True)
    print(f"\nOutput: {out}")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit("\nCancelled. Re-run to resume; finished files are skipped.")
