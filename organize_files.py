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
import logging
from logging import Logger
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
    parser.add_argument(
        "--log-file",
        type=str,
        default="organize_files.log",
        help="Path to log file (default: organize_files.log)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
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


def iter_files(base: Path, recursive: bool, category_names: Iterable[str], logger: Optional[Logger] = None) -> Iterable[Path]:
    if not recursive:
        # Use os.scandir for performance
        try:
            with os.scandir(base) as it:
                for entry in it:
                    try:
                        if entry.is_file():
                            yield Path(entry.path)
                    except PermissionError as exc:
                        if logger:
                            logger.warning("Permission denied accessing entry: %s (%s)", entry.path, exc)
        except Exception as exc:  # noqa: BLE001
            if logger:
                logger.error("Failed to scan directory %s: %s", base, exc)
        return

    for root, dirs, files in os.walk(base, onerror=(lambda e: logger.warning("Walk error: %s", e) if logger else None)):
        current_dir = Path(root)
        # Prune directories in-place for os.walk
        try:
            dirs[:] = [d for d in dirs if not should_skip_directory(current_dir / d, category_names)]
        except Exception as exc:  # noqa: BLE001
            if logger:
                logger.warning("Error pruning directories in %s: %s", current_dir, exc)
        for file_name in files:
            try:
                yield current_dir / file_name
            except Exception as exc:  # noqa: BLE001
                if logger:
                    logger.warning("Error yielding file in %s: %s", current_dir, exc)


def plan_moves(
    base: Path,
    files: Iterable[Path],
    extension_to_category: Dict[str, str],
    enabled_categories: List[str],
) -> List[MovePlan]:
    plans: List[MovePlan] = []
    enabled_set = set(enabled_categories)
    for file_path in files:
        # Skip if it's already inside a category folder under base
        try:
            relative = file_path.relative_to(base)
        except ValueError:
            # Not under base; skip
            continue

        if relative.parts and (relative.parts[0] in enabled_set or relative.parts[0] == "Others"):
            continue

        category = determine_category(file_path, extension_to_category)
        destination = (base / category) / file_path.name
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


def execute_moves(plans: Iterable[MovePlan], dry_run: bool, logger: Optional[Logger] = None) -> Tuple[int, int]:
    moved = 0
    skipped = 0
    for plan in plans:
        src = plan.source
        dst = plan.destination
        if dry_run:
            message = f"DRY-RUN: Would move '{src}' -> '{dst}' ({plan.reason})"
            if logger:
                logger.info(message)
            else:
                print(message)
            continue
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            if logger:
                logger.info("Moved: '%s' -> '%s'", src, dst)
            else:
                print(f"Moved: '{src}' -> '{dst}'")
            moved += 1
        except Exception as exc:  # noqa: BLE001
            if logger:
                logger.error("Skip: '%s' -> '%s' due to error: %s", src, dst, exc)
            else:
                print(f"Skip: '{src}' -> '{dst}' due to error: {exc}")
            skipped += 1
    return moved, skipped


def setup_logger(log_file: str, log_level: str) -> Logger:
    logger = logging.getLogger("organize_files")
    if logger.handlers:
        # Reuse existing logger
        return logger

    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    try:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as exc:  # noqa: BLE001
        # Fall back to console-only logging
        logger.warning("Could not open log file '%s': %s", log_file, exc)

    return logger


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.log_file, args.log_level)
    try:
        base = Path(args.path).expanduser().resolve()

        if not base.exists() or not base.is_dir():
            logger.error("Path does not exist or is not a directory: %s", base)
            raise SystemExit(2)

        enabled_categories = [c.strip() for c in args.categories.split(",") if c.strip()]
        extension_to_category = build_extension_to_category(enabled_categories)

        logger.info("Scanning: %s", base)
        logger.info(
            "Categories: %s",
            ", ".join(sorted(set(enabled_categories + ["Others"]))),
        )
        logger.info("Mode: %s | Recursive: %s", "DRY-RUN" if args.dry_run else "EXECUTE", args.recursive)

        files = list(iter_files(base, recursive=args.recursive, category_names=enabled_categories + ["Others"], logger=logger))
        plans = plan_moves(base, files, extension_to_category, enabled_categories)

        logger.info("Found %d file(s); %d to move.", len(files), len(plans))
        moved, skipped = execute_moves(plans, dry_run=args.dry_run, logger=logger)

        if args.dry_run:
            logger.info("DRY-RUN complete. Planned moves: %d", len(plans))
        else:
            logger.info("Done. Moved: %d, Skipped: %d", moved, skipped)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error: %s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()


