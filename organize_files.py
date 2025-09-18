#!/usr/bin/env python3
"""
Organize files in a directory by category based on file extensions.

Features:
- Classifies into Documents, Images, Videos, Audio, Archives, Code, and Others
- Creates destination folders dynamically
- Safe moves with collision handling
- Dry-run mode to preview changes

Usage:
  python organize_files.py --path C:\\path\\to\\folder --dry-run
  python organize_files.py --path .
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# Default categories and their known extensions
DEFAULT_CATEGORY_EXTENSIONS: Dict[str, Tuple[str, ...]] = {
    "Documents": (
        ".pdf", ".doc", ".docx", ".txt", ".rtf", ".odt", ".xls", ".xlsx",
        ".ppt", ".pptx", ".csv", ".md", ".json", ".xml", ".yaml", ".yml",
    ),
    "Images": (
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp",
        ".svg", ".heic",
    ),
    "Videos": (
        ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v",
    ),
    "Audio": (
        ".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma", ".aiff",
    ),
    "Archives": (
        ".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz",
    ),
    "Code": (
        ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cpp",
        ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".m",
        ".html", ".css", ".scss", ".sass", ".sh", ".ps1", ".bat",
    ),
}


@dataclass
class MovePlan:
    source: Path
    destination: Path
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Organize files by category")
    parser.add_argument(
        "--path",
        type=str,
        default=".",
        help="Directory to organize (default: current directory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview planned moves without changing files",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan subdirectories recursively (skips category folders)",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=",".join(DEFAULT_CATEGORY_EXTENSIONS.keys()),
        help=(
            "Comma-separated category names to enable. Unknowns map to Others. "
            "Default: Documents,Images,Videos,Audio,Archives,Code"
        ),
    )
    return parser.parse_args()


def build_extension_to_category(categories: Iterable[str]) -> Dict[str, str]:
    enabled = {c.strip(): DEFAULT_CATEGORY_EXTENSIONS.get(c.strip(), ()) for c in categories}
    mapping: Dict[str, str] = {}
    for category, exts in enabled.items():
        for ext in exts:
            mapping[ext.lower()] = category
    return mapping


def determine_category(path: Path, extension_to_category: Dict[str, str]) -> str:
    ext = path.suffix.lower()
    return extension_to_category.get(ext, "Others")


def should_skip_directory(directory: Path, category_names: Iterable[str]) -> bool:
    name = directory.name
    # Skip hidden/system directories and category folders to avoid moving within them
    hidden_or_system = name.startswith(".") or name.lower() in {"node_modules", "venv", "env", "__pycache__"}
    category_match = name in set(category_names)
    return hidden_or_system or category_match


def iter_files(base: Path, recursive: bool, category_names: Iterable[str]) -> Iterable[Path]:
    if not recursive:
        for entry in base.iterdir():
            if entry.is_file():
                yield entry
        return

    for root, dirs, files in os.walk(base):
        current_dir = Path(root)
        # Prune directories in-place for os.walk
        dirs[:] = [d for d in dirs if not should_skip_directory(current_dir / d, category_names)]
        for file_name in files:
            yield current_dir / file_name


def plan_moves(
    base: Path,
    files: Iterable[Path],
    extension_to_category: Dict[str, str],
    enabled_categories: List[str],
) -> List[MovePlan]:
    plans: List[MovePlan] = []
    for file_path in files:
        # Skip if it's already inside a category folder under base
        try:
            relative = file_path.relative_to(base)
        except ValueError:
            # Not under base; skip
            continue

        if relative.parts and relative.parts[0] in enabled_categories + ["Others"]:
            continue

        category = determine_category(file_path, extension_to_category)
        destination_dir = base / category
        destination_dir.mkdir(parents=True, exist_ok=True)

        destination = destination_dir / file_path.name
        if destination.exists():
            destination = resolve_collision(destination)
            reason = "name collision"
        else:
            reason = "categorized"

        plans.append(MovePlan(source=file_path, destination=destination, reason=reason))
    return plans


def resolve_collision(destination: Path) -> Path:
    stem = destination.stem
    suffix = destination.suffix
    parent = destination.parent
    counter = 1
    while True:
        candidate = parent / f"{stem} ({counter}){suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def execute_moves(plans: Iterable[MovePlan], dry_run: bool) -> Tuple[int, int]:
    moved = 0
    skipped = 0
    for plan in plans:
        src = plan.source
        dst = plan.destination
        if dry_run:
            print(f"DRY-RUN: Would move '{src}' -> '{dst}' ({plan.reason})")
            continue
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            print(f"Moved: '{src}' -> '{dst}'")
            moved += 1
        except Exception as exc:  # noqa: BLE001
            print(f"Skip: '{src}' -> '{dst}' due to error: {exc}")
            skipped += 1
    return moved, skipped


def main() -> None:
    args = parse_args()
    base = Path(args.path).expanduser().resolve()

    if not base.exists() or not base.is_dir():
        raise SystemExit(f"Path does not exist or is not a directory: {base}")

    enabled_categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    extension_to_category = build_extension_to_category(enabled_categories)

    print(f"Scanning: {base}")
    print(f"Categories: {', '.join(sorted(set(enabled_categories + ['Others'])))}")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'EXECUTE'} | Recursive: {args.recursive}")

    files = list(iter_files(base, recursive=args.recursive, category_names=enabled_categories + ["Others"]))
    plans = plan_moves(base, files, extension_to_category, enabled_categories)

    print(f"Found {len(files)} file(s); {len(plans)} to move.")
    moved, skipped = execute_moves(plans, dry_run=args.dry_run)

    if args.dry_run:
        print(f"DRY-RUN complete. Planned moves: {len(plans)}")
    else:
        print(f"Done. Moved: {moved}, Skipped: {skipped}")


if __name__ == "__main__":
    main()


