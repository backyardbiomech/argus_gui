#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fix_dlc_frame_names.py
======================
Fix frame-name mismatches in DLC CollectedData files.

Problem
-------
Argus may have saved index entries with a different zero-padding width than
the actual PNG files in the labeled-data folder.  For example, the PNG on
disk is ``img136.png`` (no padding) but the h5 index stores
``img0136.png`` (4-digit padding).  DLC tools will then fail to find the
images.

What this script does
---------------------
For every video subfolder found inside <labeled-data-folder>:

1. Scans the subfolder for ``*.png`` files and builds a definitive map of
   frame_number → actual_filename.
2. Opens the most recently modified ``CollectedData*.h5`` file.
3. Compares each index entry's image filename against the definitive map.
4. If any names differ, rewrites the ``*.h5`` **and** the matching ``*.csv``
   with the corrected index.

Usage
-----
    python fix_dlc_frame_names.py <labeled_data_folder>

    # Example:
    python fix_dlc_frame_names.py /path/to/project/labeled-data

Options
-------
    --dry-run   Print what would change without writing any files.
    --all-h5    Fix ALL CollectedData*.h5 files in each folder, not just the
                most recently modified one.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def scan_png_files(folder: Path) -> dict[int, str]:
    """Return a dict mapping *frame_number* → *actual PNG filename* for every
    ``*.png`` found directly inside *folder*."""
    frame_map: dict[int, str] = {}
    for f in sorted(folder.glob("*.png")):
        m = re.search(r"(\d+)", f.stem)
        if m:
            frame_num = int(m.group(1))
            frame_map[frame_num] = f.name
    return frame_map


def fix_index_entry(
    idx,
    actual_filenames: dict[int, str],
) -> tuple:
    """Return ``(new_index_entry, changed)`` for a single index entry *idx*.

    *idx* may be either a tuple (MultiIndex) or a plain path string.
    """
    # Extract image filename component
    if isinstance(idx, tuple):
        img_name = idx[-1]
    else:
        img_name = str(idx).replace("\\", "/").split("/")[-1]

    m = re.search(r"(\d+)", img_name)
    if not m:
        return idx, False

    frame_num = int(m.group(1))

    if frame_num not in actual_filenames:
        # No PNG on disk for this frame – leave unchanged
        return idx, False

    correct_name = actual_filenames[frame_num]
    if correct_name == img_name:
        return idx, False  # Already correct

    # Build corrected entry preserving original format
    if isinstance(idx, tuple):
        parts = list(idx)
        parts[-1] = correct_name
        return tuple(parts), True
    else:
        idx_str = str(idx)
        if "\\" in idx_str:
            parts = idx_str.split("\\")
            parts[-1] = correct_name
            return "\\".join(parts), True
        else:
            parts = idx_str.split("/")
            parts[-1] = correct_name
            return "/".join(parts), True


def fix_h5_file(
    h5_path: Path,
    video_folder: Path,
    dry_run: bool = False,
) -> bool:
    """Fix frame names in *h5_path* to match PNGs in *video_folder*.

    Returns ``True`` if any changes were (or would be) made.
    """
    frame_map = scan_png_files(video_folder)
    if not frame_map:
        print(f"    [skip] No PNG files found in {video_folder.name}")
        return False

    df = pd.read_hdf(h5_path)

    new_index_entries = []
    changed_count = 0

    for idx in df.index:
        new_idx, changed = fix_index_entry(idx, frame_map)
        new_index_entries.append(new_idx)
        if changed:
            # Extract old/new filename for the log message
            old_name = idx[-1] if isinstance(idx, tuple) else str(idx).replace("\\", "/").split("/")[-1]
            new_name = new_idx[-1] if isinstance(new_idx, tuple) else str(new_idx).replace("\\", "/").split("/")[-1]
            frame_str = re.search(r"(\d+)", old_name).group(1)
            print(f"    frame {frame_str:>6}: {old_name!r:30s} -> {new_name!r}")
            changed_count += 1

    if changed_count == 0:
        print(f"    [ok]  No changes needed in {h5_path.name}")
        return False

    print(f"    {changed_count} frame name(s) will be updated")

    if dry_run:
        print(f"    [dry-run] Would save {h5_path.name} and matching .csv")
        return True

    # Apply new index
    if isinstance(df.index, pd.MultiIndex):
        df.index = pd.MultiIndex.from_tuples(new_index_entries, names=df.index.names)
    else:
        df.index = pd.Index(new_index_entries, name=df.index.name)

    # Save h5
    df.to_hdf(h5_path, key="df_with_missing", mode="w")

    # Save matching csv (same stem, .csv extension)
    csv_path = h5_path.with_suffix(".csv")
    df.to_csv(csv_path)

    print(f"    Saved: {h5_path.name}")
    print(f"    Saved: {csv_path.name}")
    return True


def process_video_folder(
    video_folder: Path,
    dry_run: bool = False,
    all_h5: bool = False,
) -> None:
    """Process a single video subfolder."""
    print(f"\n  [{video_folder.name}]")

    h5_files = list(video_folder.glob("CollectedData*.h5"))
    if not h5_files:
        print(f"    [skip] No CollectedData*.h5 files found")
        return

    if all_h5:
        targets = sorted(h5_files, key=lambda p: p.stat().st_mtime, reverse=True)
        print(f"    Found {len(targets)} CollectedData file(s) – processing all")
    else:
        # Use only the most recently modified file
        targets = [max(h5_files, key=lambda p: p.stat().st_mtime)]
        if len(h5_files) > 1:
            print(f"    Found {len(h5_files)} CollectedData files – using most recent: {targets[0].name}")
        else:
            print(f"    File: {targets[0].name}")

    for h5_path in targets:
        try:
            fix_h5_file(h5_path, video_folder, dry_run=dry_run)
        except Exception as exc:
            print(f"    [error] {h5_path.name}: {exc}")
            import traceback
            traceback.print_exc()


def fix_labeled_data_folder(
    labeled_data_path: str | Path,
    dry_run: bool = False,
    all_h5: bool = False,
) -> None:
    """Main entry point: iterate over all video subfolders in *labeled_data_path*."""
    labeled_data_path = Path(labeled_data_path)

    if not labeled_data_path.exists():
        print(f"Error: path does not exist: {labeled_data_path}")
        sys.exit(1)

    subfolders = sorted(
        [f for f in labeled_data_path.iterdir() if f.is_dir()]
    )

    if not subfolders:
        print(f"No subfolders found in {labeled_data_path}")
        return

    mode_str = " (DRY RUN)" if dry_run else ""
    print(f"Fixing frame names in: {labeled_data_path}{mode_str}")
    print(f"Video folders found: {len(subfolders)}")

    for folder in subfolders:
        process_video_folder(folder, dry_run=dry_run, all_h5=all_h5)

    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fix DLC CollectedData frame names to match actual PNG files on disk."
        )
    )
    parser.add_argument(
        "labeled_data_folder",
        help="Path to the DLC labeled-data folder (contains one subfolder per video).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print what would change without writing any files.",
    )
    parser.add_argument(
        "--all-h5",
        action="store_true",
        default=False,
        help=(
            "Fix ALL CollectedData*.h5 files in each video folder, not just the "
            "most recently modified one."
        ),
    )

    args = parser.parse_args()
    fix_labeled_data_folder(
        args.labeled_data_folder,
        dry_run=args.dry_run,
        all_h5=args.all_h5,
    )


if __name__ == "__main__":
    main()
