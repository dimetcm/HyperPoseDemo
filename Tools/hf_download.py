#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["huggingface_hub"]
# ///
"""Download assets from a Hugging Face repo (model, dataset or space).

Examples:
    uv run hf_download.py someuser/some-repo
    uv run hf_download.py someuser/mocap-data --type dataset --include "*.fbx" "*.bvh"
    uv run hf_download.py someuser/some-repo --file path/in/repo/clip.fbx
    uv run hf_download.py someuser/private-repo --token hf_xxx --out ../Assets/HF

`uv run` installs huggingface_hub automatically (or: pip install -U huggingface_hub).
Private/gated repos need a token: pass --token, set HF_TOKEN, or run `hf auth login`.
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
except ImportError:
    sys.exit("huggingface_hub is not installed. Run: pip install -U huggingface_hub")

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "Assets" / "HuggingFace"


def parse_args():
    p = argparse.ArgumentParser(description="Download assets from Hugging Face.")
    p.add_argument("repo_id", help="Repo id, e.g. 'user/repo'")
    p.add_argument("--type", dest="repo_type", default="model",
                   choices=["model", "dataset", "space"], help="Repo type (default: model)")
    p.add_argument("--revision", default=None, help="Branch, tag or commit (default: main)")
    p.add_argument("--file", action="append", default=[],
                   help="Download only this file (repeatable)")
    p.add_argument("--include", nargs="+", default=None,
                   help="Glob patterns to include, e.g. '*.fbx' 'anims/*'")
    p.add_argument("--exclude", nargs="+", default=None,
                   help="Glob patterns to exclude")
    p.add_argument("--out", type=Path, default=None,
                   help=f"Output folder (default: {DEFAULT_OUT}/<repo>)")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                   help="HF access token (default: $HF_TOKEN or cached login)")
    p.add_argument("--workers", type=int, default=8, help="Parallel downloads (default: 8)")
    p.add_argument("--list", action="store_true", help="Only list files in the repo, don't download")
    return p.parse_args()


def main():
    args = parse_args()
    out = args.out or DEFAULT_OUT / args.repo_id.replace("/", "__")
    common = dict(repo_id=args.repo_id, repo_type=args.repo_type,
                  revision=args.revision, token=args.token)

    if args.list:
        for f in sorted(list_repo_files(**common)):
            print(f)
        return

    out.mkdir(parents=True, exist_ok=True)

    if args.file:
        for f in args.file:
            path = hf_hub_download(filename=f, local_dir=out, **common)
            print(f"Downloaded: {path}")
    else:
        path = snapshot_download(local_dir=out, allow_patterns=args.include,
                                 ignore_patterns=args.exclude,
                                 max_workers=args.workers, **common)
        print(f"Downloaded to: {path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nCancelled. Re-run to resume; finished files are skipped.")
    except Exception as e:
        sys.exit(f"Error: {e}")
