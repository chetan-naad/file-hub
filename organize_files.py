#!/usr/bin/env python3
"""
Smart File Organizer - Phase 3
Advanced file organization with smart features.

Features:
- Classifies into Documents, Images, Videos, Audio, Archives, Code, and Others
- Creates destination folders dynamically
- Safe moves with collision handling
- Dry-run mode to preview changes
- Smart categorization (date, size, hybrid)
- Duplicate finder using MD5 hashing
- Scheduler for auto-run
- Advanced file organization
- Interactive duplicate removal
- Comprehensive statistics

Usage:
  python organize_files.py --path C:\\path\\to\\folder --dry-run
  python organize_files.py --path . --mode hybrid
  python organize_files.py --path . --find-duplicates
  python organize_files.py --path . --schedule daily --time 02:00
  python organize_files.py --interactive
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import logging
from logging import Logger
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any
import json

try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False

# Try to import drag-and-drop support
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False


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


class SmartFileOrganizer:
    """Thin wrapper exposing programmatic API with optional callbacks."""

    def __init__(self, logger: Optional[Logger] = None):
        self.logger: Logger = logger or logging.getLogger("organize_files.SmartFileOrganizer")
        if not self.logger.handlers:
            # Default console-only logger if not configured by caller
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
            self.logger.addHandler(handler)
        self._progress_cb = None
        self._status_cb = None

    def set_callbacks(self, progress_callback=None, status_callback=None) -> None:
        self._progress_cb = progress_callback
        self._status_cb = status_callback

    def _progress(self, value: float) -> None:
        try:
            if self._progress_cb:
                self._progress_cb(value)
        except Exception:  # noqa: BLE001
            pass

    def _status(self, message: str) -> None:
        if self._status_cb:
            try:
                self._status_cb(message)
            except Exception:  # noqa: BLE001
                pass
        self.logger.info(message)

    def organize_extension(self, directory: Path, categories: Optional[List[str]] = None, dry_run: bool = False) -> Tuple[int, int]:
        base = Path(directory)
        cats = categories or list(DEFAULT_CATEGORY_EXTENSIONS.keys())
        ext_to_cat = build_extension_to_category(cats)
        self._status(f"Organizing by extension in: {base}")
        files = list(iter_files(base, recursive=False, category_names=cats + ["Others"], logger=self.logger))
        plans = plan_moves(base, files, ext_to_cat, cats)
        total = max(1, len(plans))
        moved = 0
        skipped = 0
        if dry_run:
            for idx, p in enumerate(plans, 1):
                self.logger.info("DRY-RUN: Would move '%s' -> '%s' (%s)", p.source, p.destination, p.reason)
                self._progress(idx * 100.0 / total)
            self._status(f"DRY-RUN complete. Planned moves: {len(plans)}")
            return 0, 0
        for idx, p in enumerate(plans, 1):
            try:
                p.destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p.source), str(p.destination))
                self.logger.info("Moved: '%s' -> '%s'", p.source, p.destination)
                moved += 1
            except Exception as exc:  # noqa: BLE001
                self.logger.error("Skip: '%s' -> '%s' due to error: %s", p.source, p.destination, exc)
                skipped += 1
            finally:
                self._progress(idx * 100.0 / total)
        self._status(f"Done. Moved: {moved}, Skipped: {skipped}")
        return moved, skipped

    def organize_date(self, directory: Path, date_format: str = "year_month") -> int:
        self._status(f"Organizing by date in: {directory}")
        return organize_by_date(Path(directory), date_format, logger=self.logger)

    def organize_size(self, directory: Path) -> int:
        self._status(f"Organizing by size in: {directory}")
        return organize_by_size(Path(directory), logger=self.logger)

    def organize_hybrid(self, directory: Path, categories: Optional[List[str]] = None) -> int:
        cats = categories or list(DEFAULT_CATEGORY_EXTENSIONS.keys())
        mapping = build_extension_to_category(cats)
        self._status(f"Hybrid organization in: {directory}")
        return organize_hybrid(Path(directory), mapping, logger=self.logger)

    def find_duplicates_in(self, directory: Path) -> List[DuplicateGroup]:
        self._status(f"Scanning for duplicates in: {directory}")
        return find_duplicates(Path(directory), logger=self.logger)

    def get_stats(self, directory: Path) -> Dict:
        self._status(f"Collecting directory stats in: {directory}")
        return get_directory_stats(Path(directory), logger=self.logger)


class EnhancedSmartFileOrganizer:
    """Enhanced file organizer with all features"""
    
    def __init__(self, logger: Optional[Logger] = None):
        self.logger: Logger = logger or logging.getLogger("organize_files.EnhancedSmartFileOrganizer")
        if not self.logger.handlers:
            # Default console-only logger if not configured by caller
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
            self.logger.addHandler(handler)
        self._progress_cb = None
        self._status_cb = None

    def set_callbacks(self, progress_callback=None, status_callback=None) -> None:
        self._progress_cb = progress_callback
        self._status_cb = status_callback

    def _progress(self, value: float) -> None:
        try:
            if self._progress_cb:
                self._progress_cb(value)
        except Exception:
            pass

    def _status(self, message: str) -> None:
        if self._status_cb:
            try:
                self._status_cb(message)
            except Exception:
                pass
        self.logger.info(message)

    def organize_extension(self, directory: Path, categories: Optional[List[str]] = None, dry_run: bool = False) -> Tuple[int, int]:
        base = Path(directory)
        cats = categories or list(DEFAULT_CATEGORY_EXTENSIONS.keys())
        ext_to_cat = self.build_extension_to_category(cats)
        self._status(f"Organizing by extension in: {base}")
        files = list(self.iter_files(base, recursive=False, category_names=cats + ["Others"]))
        plans = self.plan_moves(base, files, ext_to_cat, cats)
        total = max(1, len(plans))
        moved = 0
        skipped = 0
        
        if dry_run:
            for idx, p in enumerate(plans, 1):
                self.logger.info("DRY-RUN: Would move '%s' -> '%s' (%s)", p.source, p.destination, p.reason)
                self._progress(idx * 100.0 / total)
            self._status(f"DRY-RUN complete. Planned moves: {len(plans)}")
            return 0, 0
        
        for idx, p in enumerate(plans, 1):
            try:
                p.destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p.source), str(p.destination))
                self.logger.info("Moved: '%s' -> '%s'", p.source, p.destination)
                moved += 1
            except Exception as exc:
                self.logger.error("Skip: '%s' -> '%s' due to error: %s", p.source, p.destination, exc)
                skipped += 1
            finally:
                self._progress(idx * 100.0 / total)
        
        self._status(f"Done. Moved: {moved}, Skipped: {skipped}")
        return moved, skipped

    def find_duplicates_in(self, directory: Path) -> List[List[Path]]:
        self._status(f"Scanning for duplicates in: {directory}")
        hash_map: Dict[str, List[Path]] = defaultdict(list)
        duplicates: List[List[Path]] = []
        file_count = 0
        
        try:
            # Scan all files and calculate hashes
            for root, _, files in os.walk(directory):
                for file in files:
                    file_path = Path(root) / file
                    
                    # Skip system files and very small files
                    if file_path.stat().st_size < 10:  # Skip files smaller than 10 bytes
                        continue
                    
                    file_hash = self.get_file_hash(file_path)
                    if file_hash:
                        hash_map[file_hash].append(file_path)
                        file_count += 1
                        
                        # Progress indication
                        if file_count % 100 == 0:
                            self.logger.info(f"Processed {file_count} files...")
            
            # Find duplicates
            total_duplicate_files = 0
            for file_hash, paths in hash_map.items():
                if len(paths) > 1:
                    duplicates.append(paths)
                    total_duplicate_files += len(paths)
            
            # Statistics
            self.logger.info(f"Duplicate scan complete")
            self.logger.info(f"Files scanned: {file_count}")
            self.logger.info(f"Duplicate sets found: {len(duplicates)}")
            self.logger.info(f"Total duplicate files: {total_duplicate_files}")
            
            return duplicates
            
        except Exception as e:
            self.logger.error(f"Error during duplicate scan: {e}")
            return []

    def get_file_hash(self, file_path: Path, chunk_size: int = 8192) -> Optional[str]:
        """Generate MD5 hash for duplicate detection"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                # Read file in chunks to handle large files efficiently
                while chunk := f.read(chunk_size):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.error(f"Error hashing {file_path}: {e}")
            return None

    def get_stats(self, directory: Path) -> Dict:
        self._status(f"Collecting directory stats in: {directory}")
        stats = {
            'total_files': 0,
            'total_size': 0,
            'file_types': Counter(),
            'size_distribution': {'small': 0, 'medium': 0, 'large': 0},
            'oldest_file': None,
            'newest_file': None,
            'categories': Counter()
        }
        
        extension_mapping = self.build_extension_to_category(DEFAULT_CATEGORY_EXTENSIONS.keys())
        
        try:
            for root, _, files in os.walk(directory):
                for file in files:
                    file_path = Path(root) / file
                    file_stat = file_path.stat()
                    
                    stats['total_files'] += 1
                    stats['total_size'] += file_stat.st_size
                    stats['file_types'][file_path.suffix.lower()] += 1
                    
                    # Category
                    ext = file_path.suffix.lower()
                    category = extension_mapping.get(ext, "Others")
                    stats['categories'][category] += 1
                    
                    # Size distribution
                    if file_stat.st_size < 1024 * 1024:
                        stats['size_distribution']['small'] += 1
                    elif file_stat.st_size < 100 * 1024 * 1024:
                        stats['size_distribution']['medium'] += 1
                    else:
                        stats['size_distribution']['large'] += 1
                    
                    # Track oldest and newest files
                    creation_time = file_stat.st_ctime
                    if stats['oldest_file'] is None or creation_time < stats['oldest_file'][1]:
                        stats['oldest_file'] = (str(file_path), creation_time)
                    if stats['newest_file'] is None or creation_time > stats['newest_file'][1]:
                        stats['newest_file'] = (str(file_path), creation_time)
        
        except Exception as e:
            self.logger.error(f"Error getting directory stats: {e}")
        
        return stats

    # Helper methods
    def build_extension_to_category(self, categories: Iterable[str]) -> Dict[str, str]:
        enabled = {c.strip(): DEFAULT_CATEGORY_EXTENSIONS.get(c.strip(), ()) for c in categories}
        mapping: Dict[str, str] = {}
        for category, exts in enabled.items():
            for ext in exts:
                mapping[ext.lower()] = category
        return mapping

    def determine_category(self, path: Path, extension_to_category: Dict[str, str]) -> str:
        ext = path.suffix.lower()
        return extension_to_category.get(ext, "Others")

    def iter_files(self, base: Path, recursive: bool, category_names: Iterable[str]) -> Iterable[Path]:
        if not recursive:
            # Use os.scandir for performance
            try:
                with os.scandir(base) as it:
                    for entry in it:
                        try:
                            if entry.is_file():
                                yield Path(entry.path)
                        except PermissionError as exc:
                            self.logger.warning("Permission denied accessing entry: %s (%s)", entry.path, exc)
            except Exception as exc:
                self.logger.error("Failed to scan directory %s: %s", base, exc)
            return

        for root, dirs, files in os.walk(base, onerror=lambda e: self.logger.warning("Walk error: %s", e)):
            current_dir = Path(root)
            # Prune directories in-place for os.walk
            try:
                dirs[:] = [d for d in dirs if not self.should_skip_directory(current_dir / d, category_names)]
            except Exception as exc:
                self.logger.warning("Error pruning directories in %s: %s", current_dir, exc)
            for file_name in files:
                try:
                    yield current_dir / file_name
                except Exception as exc:
                    self.logger.warning("Error yielding file in %s: %s", current_dir, exc)

    def should_skip_directory(self, directory: Path, category_names: Iterable[str]) -> bool:
        name = directory.name
        # Skip hidden/system directories and category folders to avoid moving within them
        hidden_or_system = name.startswith(".") or name.lower() in {"node_modules", "venv", "env", "__pycache__"}
        category_match = name in set(category_names)
        return hidden_or_system or category_match

    def plan_moves(self, base: Path, files: Iterable[Path], extension_to_category: Dict[str, str], enabled_categories: List[str]) -> List[MovePlan]:
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

            category = self.determine_category(file_path, extension_to_category)
            destination = (base / category) / file_path.name
            
            file_size = 0
            try:
                file_size = file_path.stat().st_size
            except:
                pass
            
            if destination.exists():
                destination = self.resolve_collision(destination)
                reason = "name collision"
            else:
                reason = "categorized"

            plans.append(MovePlan(source=file_path, destination=destination, reason=reason, size=file_size, category=category))
        return plans

    def resolve_collision(self, destination: Path) -> Path:
        stem = destination.stem
        suffix = destination.suffix
        parent = destination.parent
        counter = 1
        while True:
            candidate = parent / f"{stem} ({counter}){suffix}"
            if not candidate.exists():
                return candidate
            counter += 1


# BasicOrganizerGUI removed – only Enhanced GUI is supported

@dataclass
class MovePlan:
    source: Path
    destination: Path
    reason: str
    size: int = 0
    category: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source': str(self.source),
            'destination': str(self.destination),
            'reason': self.reason,
            'size': self.size,
            'category': self.category
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MovePlan':
        return cls(
            source=Path(data['source']),
            destination=Path(data['destination']),
            reason=data['reason'],
            size=data.get('size', 0),
            category=data.get('category', '')
        )


@dataclass
class UndoOperation:
    operation_type: str  # 'organize', 'move', 'delete'
    timestamp: datetime
    plans: List[MovePlan]
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'operation_type': self.operation_type,
            'timestamp': self.timestamp.isoformat(),
            'plans': [p.to_dict() for p in self.plans],
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UndoOperation':
        return cls(
            operation_type=data['operation_type'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            plans=[MovePlan.from_dict(p) for p in data['plans']],
            description=data['description']
        )


@dataclass
class DuplicateGroup:
    files: List[Path]
    hash_value: str
    total_size: int


class ProgressTracker:
    """Enhanced progress tracking with detailed statistics"""
    
    def __init__(self, callback=None):
        self.callback = callback
        self.reset()
    
    def reset(self):
        self.total_files = 0
        self.processed_files = 0
        self.total_size = 0
        self.processed_size = 0
        self.errors = 0
        self.category_counts = Counter()
        self.start_time = time.time()
        self.current_file = ""
        
    def update(self, current_file: str = "", file_size: int = 0, category: str = "", error: bool = False):
        self.processed_files += 1
        self.processed_size += file_size
        self.current_file = current_file
        if category:
            self.category_counts[category] += 1
        if error:
            self.errors += 1
        
        if self.callback:
            self.callback(self.get_progress_info())
    
    def set_total(self, total_files: int, total_size: int = 0):
        self.total_files = total_files
        self.total_size = total_size
    
    def get_progress_info(self) -> Dict[str, Any]:
        elapsed_time = time.time() - self.start_time
        progress_percent = (self.processed_files / max(1, self.total_files)) * 100
        
        # Estimate remaining time
        if self.processed_files > 0:
            avg_time_per_file = elapsed_time / self.processed_files
            remaining_files = self.total_files - self.processed_files
            eta = remaining_files * avg_time_per_file
        else:
            eta = 0
        
        return {
            'progress_percent': progress_percent,
            'processed_files': self.processed_files,
            'total_files': self.total_files,
            'processed_size': self.processed_size,
            'total_size': self.total_size,
            'current_file': self.current_file,
            'elapsed_time': elapsed_time,
            'eta': eta,
            'errors': self.errors,
            'category_counts': dict(self.category_counts)
        }


class UndoRedoManager:
    """Manages undo/redo operations for file organization"""
    
    def __init__(self, max_operations: int = 10):
        self.max_operations = max_operations
        self.operations: List[UndoOperation] = []
        self.current_index = -1
        self.undo_file = Path("file_organizer_undo.json")
        self.load_operations()
    
    def add_operation(self, operation: UndoOperation):
        # Remove any operations after current index (when doing new operations after undo)
        self.operations = self.operations[:self.current_index + 1]
        
        # Add new operation
        self.operations.append(operation)
        self.current_index += 1
        
        # Limit the number of stored operations
        if len(self.operations) > self.max_operations:
            self.operations.pop(0)
            self.current_index -= 1
        
        self.save_operations()
    
    def can_undo(self) -> bool:
        return self.current_index >= 0
    
    def can_redo(self) -> bool:
        return self.current_index < len(self.operations) - 1
    
    def undo(self) -> Optional[UndoOperation]:
        if not self.can_undo():
            return None
        
        operation = self.operations[self.current_index]
        self.current_index -= 1
        self.save_operations()
        return operation
    
    def redo(self) -> Optional[UndoOperation]:
        if not self.can_redo():
            return None
        
        self.current_index += 1
        operation = self.operations[self.current_index]
        self.save_operations()
        return operation
    
    def save_operations(self):
        try:
            data = {
                'operations': [op.to_dict() for op in self.operations],
                'current_index': self.current_index
            }
            with open(self.undo_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass  # Silently fail if can't save
    
    def load_operations(self):
        try:
            if self.undo_file.exists():
                with open(self.undo_file, 'r') as f:
                    data = json.load(f)
                self.operations = [UndoOperation.from_dict(op) for op in data['operations']]
                self.current_index = data['current_index']
        except Exception:
            self.operations = []
            self.current_index = -1


def get_file_hash(file_path: Path, chunk_size: int = 8192) -> Optional[str]:
    """
    Generate MD5 hash for duplicate detection
    Optimized for large files with chunked reading
    """
    try:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            # Read file in chunks to handle large files efficiently
            while chunk := f.read(chunk_size):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"Error hashing {file_path}: {e}")
        return None


def find_duplicates(directory: Path, logger: Optional[Logger] = None) -> List[DuplicateGroup]:
    """
    Find duplicate files using MD5 hashing
    Returns list of duplicate file groups
    """
    if logger:
        logger.info(f"🔍 Starting duplicate scan in: {directory}")
    else:
        print(f"🔍 Starting duplicate scan in: {directory}")
    
    hash_map: Dict[str, List[Path]] = defaultdict(list)
    duplicates: List[DuplicateGroup] = []
    file_count = 0
    
    try:
        # Scan all files and calculate hashes
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = Path(root) / file
                
                # Skip system files and very small files
                if file_path.stat().st_size < 10:  # Skip files smaller than 10 bytes
                    continue
                
                file_hash = get_file_hash(file_path)
                if file_hash:
                    hash_map[file_hash].append(file_path)
                    file_count += 1
                    
                    # Progress indication
                    if file_count % 100 == 0:
                        if logger:
                            logger.info(f"Processed {file_count} files...")
                        else:
                            print(f"Processed {file_count} files...")
        
        # Find duplicates
        total_duplicate_files = 0
        for file_hash, paths in hash_map.items():
            if len(paths) > 1:
                total_size = sum(path.stat().st_size for path in paths)
                duplicates.append(DuplicateGroup(
                    files=paths,
                    hash_value=file_hash,
                    total_size=total_size
                ))
                total_duplicate_files += len(paths)
        
        # Statistics
        if logger:
            logger.info(f"✓ Duplicate scan complete")
            logger.info(f"📊 Files scanned: {file_count}")
            logger.info(f"🔄 Duplicate sets found: {len(duplicates)}")
            logger.info(f"📁 Total duplicate files: {total_duplicate_files}")
        else:
            print(f"✓ Duplicate scan complete")
            print(f"📊 Files scanned: {file_count}")
            print(f"🔄 Duplicate sets found: {len(duplicates)}")
            print(f"📁 Total duplicate files: {total_duplicate_files}")
        
        return duplicates
        
    except Exception as e:
        if logger:
            logger.error(f"Error during duplicate scan: {e}")
        else:
            print(f"Error during duplicate scan: {e}")
        return []


def remove_duplicates_interactive(directory: Path, logger: Optional[Logger] = None) -> Tuple[int, int]:
    """
    Interactive duplicate removal
    Shows duplicates and asks user which ones to keep
    Returns (removed_count, saved_space_bytes)
    """
    duplicates = find_duplicates(directory, logger)
    
    if not duplicates:
        if logger:
            logger.info("No duplicates found!")
        else:
            print("No duplicates found!")
        return 0, 0
    
    removed_count = 0
    saved_space = 0
    
    for i, duplicate_group in enumerate(duplicates, 1):
        print(f"\n🔄 Duplicate Set {i}:")
        for j, file_path in enumerate(duplicate_group.files):
            file_size = file_path.stat().st_size
            print(f"  {j+1}. {file_path} ({file_size / 1024:.1f} KB)")
        
        try:
            choice = input(f"Keep which file? (1-{len(duplicate_group.files)}, 'a' for all, 's' for skip): ").strip()
            
            if choice.lower() == 's':
                continue
            elif choice.lower() == 'a':
                continue
            else:
                keep_index = int(choice) - 1
                if 0 <= keep_index < len(duplicate_group.files):
                    # Remove all except the chosen one
                    for j, file_path in enumerate(duplicate_group.files):
                        if j != keep_index:
                            file_size = file_path.stat().st_size
                            file_path.unlink()  # Delete file
                            removed_count += 1
                            saved_space += file_size
                            if logger:
                                logger.info(f"🗑️ Removed duplicate: {file_path}")
                            else:
                                print(f"🗑️ Removed duplicate: {file_path}")
            
        except (ValueError, IndexError):
            print("Invalid choice, skipping this set")
            continue
    
    if logger:
        logger.info(f"✓ Duplicate removal complete")
        logger.info(f"🗑️ Files removed: {removed_count}")
        logger.info(f"💾 Space saved: {saved_space / (1024*1024):.2f} MB")
    else:
        print(f"✓ Duplicate removal complete")
        print(f"🗑️ Files removed: {removed_count}")
        print(f"💾 Space saved: {saved_space / (1024*1024):.2f} MB")
    
    return removed_count, saved_space


def organize_by_date(directory: Path, date_format: str = "year_month", logger: Optional[Logger] = None) -> int:
    """
    Organize files by creation date
    Formats: 'year_month' (2024/January), 'year_only' (2024), 'full_date' (2024/01/15)
    """
    if logger:
        logger.info(f"📅 Organizing by date in: {directory}")
    else:
        print(f"📅 Organizing by date in: {directory}")
    
    moved_count = 0
    
    try:
        for file_path in directory.iterdir():
            if file_path.is_file():
                try:
                    # Get file creation time
                    creation_time = file_path.stat().st_ctime
                    created_date = datetime.fromtimestamp(creation_time)
                    
                    # Generate folder name based on format
                    if date_format == "year_month":
                        folder_name = f"{created_date.year}/{created_date.strftime('%B')}"
                    elif date_format == "year_only":
                        folder_name = str(created_date.year)
                    elif date_format == "full_date":
                        folder_name = created_date.strftime("%Y/%m/%d")
                    else:
                        folder_name = created_date.strftime("%Y/%B")
                    
                    # Create target directory
                    target_dir = directory / folder_name
                    target_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Move file
                    destination = target_dir / file_path.name
                    if destination.exists():
                        # Handle naming conflicts
                        counter = 1
                        stem = file_path.stem
                        suffix = file_path.suffix
                        while destination.exists():
                            destination = target_dir / f"{stem}_{counter}{suffix}"
                            counter += 1
                    
                    shutil.move(str(file_path), str(destination))
                    moved_count += 1
                    if logger:
                        logger.info(f"📅 Moved {file_path.name} → {folder_name}/")
                    else:
                        print(f"📅 Moved {file_path.name} → {folder_name}/")
                    
                except Exception as e:
                    if logger:
                        logger.error(f"Error organizing {file_path.name} by date: {e}")
                    else:
                        print(f"Error organizing {file_path.name} by date: {e}")
        
        if logger:
            logger.info(f"✓ Date organization complete. Moved {moved_count} files")
        else:
            print(f"✓ Date organization complete. Moved {moved_count} files")
        return moved_count
        
    except Exception as e:
        if logger:
            logger.error(f"Error in date organization: {e}")
        else:
            print(f"Error in date organization: {e}")
        return 0


def organize_by_size(directory: Path, logger: Optional[Logger] = None) -> int:
    """
    Organize files by size categories:
    - Small: < 1MB
    - Medium: 1MB - 100MB  
    - Large: > 100MB
    """
    if logger:
        logger.info(f"📏 Organizing by size in: {directory}")
    else:
        print(f"📏 Organizing by size in: {directory}")
    
    moved_count = 0
    size_stats = defaultdict(int)
    
    try:
        for file_path in directory.iterdir():
            if file_path.is_file():
                try:
                    file_size = file_path.stat().st_size
                    
                    # Determine size category
                    if file_size < 1024 * 1024:  # < 1MB
                        category = "Small_Files"
                        size_label = f"{file_size / 1024:.1f}KB"
                    elif file_size < 100 * 1024 * 1024:  # < 100MB
                        category = "Medium_Files" 
                        size_label = f"{file_size / (1024*1024):.1f}MB"
                    else:  # >= 100MB
                        category = "Large_Files"
                        size_label = f"{file_size / (1024*1024):.1f}MB"
                    
                    # Create target directory
                    target_dir = directory / category
                    target_dir.mkdir(exist_ok=True)
                    
                    # Move file
                    destination = target_dir / file_path.name
                    if destination.exists():
                        counter = 1
                        stem = file_path.stem
                        suffix = file_path.suffix
                        while destination.exists():
                            destination = target_dir / f"{stem}_{counter}{suffix}"
                            counter += 1
                    
                    shutil.move(str(file_path), str(destination))
                    moved_count += 1
                    size_stats[category] += 1
                    
                    if logger:
                        logger.info(f"📏 Moved {file_path.name} ({size_label}) → {category}/")
                    else:
                        print(f"📏 Moved {file_path.name} ({size_label}) → {category}/")
                    
                except Exception as e:
                    if logger:
                        logger.error(f"Error organizing {file_path.name} by size: {e}")
                    else:
                        print(f"Error organizing {file_path.name} by size: {e}")
        
        # Report statistics
        if logger:
            logger.info(f"✓ Size organization complete. Moved {moved_count} files")
            for category, count in size_stats.items():
                logger.info(f"  {category}: {count} files")
        else:
            print(f"✓ Size organization complete. Moved {moved_count} files")
            for category, count in size_stats.items():
                print(f"  {category}: {count} files")
        
        return moved_count
        
    except Exception as e:
        if logger:
            logger.error(f"Error in size organization: {e}")
        else:
            print(f"Error in size organization: {e}")
        return 0


def organize_hybrid(directory: Path, extension_to_category: Dict[str, str], logger: Optional[Logger] = None) -> int:
    """
    Hybrid organization: Group by file type, then by date within each type
    Example: Images/2024/January/photo.jpg
    """
    if logger:
        logger.info(f"🔄 Hybrid organization in: {directory}")
    else:
        print(f"🔄 Hybrid organization in: {directory}")
    
    moved_count = 0
    
    try:
        for file_path in directory.iterdir():
            if file_path.is_file():
                try:
                    # Get file type category
                    file_ext = file_path.suffix.lower()
                    category = extension_to_category.get(file_ext, "Others")
                    
                    # Get creation date
                    creation_time = file_path.stat().st_ctime
                    created_date = datetime.fromtimestamp(creation_time)
                    date_folder = f"{created_date.year}/{created_date.strftime('%B')}"
                    
                    # Create nested folder structure: Category/Year/Month
                    target_dir = directory / category / date_folder
                    target_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Move file
                    destination = target_dir / file_path.name
                    if destination.exists():
                        counter = 1
                        stem = file_path.stem
                        suffix = file_path.suffix
                        while destination.exists():
                            destination = target_dir / f"{stem}_{counter}{suffix}"
                            counter += 1
                    
                    shutil.move(str(file_path), str(destination))
                    moved_count += 1
                    
                    if logger:
                        logger.info(f"🔄 Moved {file_path.name} → {category}/{date_folder}/")
                    else:
                        print(f"🔄 Moved {file_path.name} → {category}/{date_folder}/")
                    
                except Exception as e:
                    if logger:
                        logger.error(f"Error in hybrid organization for {file_path.name}: {e}")
                    else:
                        print(f"Error in hybrid organization for {file_path.name}: {e}")
        
        if logger:
            logger.info(f"✓ Hybrid organization complete. Moved {moved_count} files")
        else:
            print(f"✓ Hybrid organization complete. Moved {moved_count} files")
        return moved_count
        
    except Exception as e:
        if logger:
            logger.error(f"Error in hybrid organization: {e}")
        else:
            print(f"Error in hybrid organization: {e}")
        return 0


def get_directory_stats(directory: Path, logger: Optional[Logger] = None) -> Dict:
    """Get comprehensive directory statistics"""
    stats = {
        'total_files': 0,
        'total_size': 0,
        'file_types': Counter(),
        'size_distribution': {'small': 0, 'medium': 0, 'large': 0},
        'oldest_file': None,
        'newest_file': None
    }
    
    try:
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = Path(root) / file
                file_stat = file_path.stat()
                
                stats['total_files'] += 1
                stats['total_size'] += file_stat.st_size
                stats['file_types'][file_path.suffix.lower()] += 1
                
                # Size distribution
                if file_stat.st_size < 1024 * 1024:
                    stats['size_distribution']['small'] += 1
                elif file_stat.st_size < 100 * 1024 * 1024:
                    stats['size_distribution']['medium'] += 1
                else:
                    stats['size_distribution']['large'] += 1
                
                # Track oldest and newest files
                creation_time = file_stat.st_ctime
                if stats['oldest_file'] is None or creation_time < stats['oldest_file'][1]:
                    stats['oldest_file'] = (str(file_path), creation_time)
                if stats['newest_file'] is None or creation_time > stats['newest_file'][1]:
                    stats['newest_file'] = (str(file_path), creation_time)
    
    except Exception as e:
        if logger:
            logger.error(f"Error getting directory stats: {e}")
        else:
            print(f"Error getting directory stats: {e}")
    
    return stats


def print_directory_stats(directory: Path, logger: Optional[Logger] = None) -> None:
    """Print comprehensive directory statistics"""
    stats = get_directory_stats(directory, logger)
    
    if logger:
        logger.info("📊 Directory Statistics:")
        logger.info(f"Total files: {stats['total_files']:,}")
        logger.info(f"Total size: {stats['total_size'] / (1024*1024*1024):.2f} GB")
        logger.info(f"File types: {len(stats['file_types'])}")
        logger.info("Size distribution:")
        logger.info(f"  Small (<1MB): {stats['size_distribution']['small']}")
        logger.info(f"  Medium (1-100MB): {stats['size_distribution']['medium']}")
        logger.info(f"  Large (>100MB): {stats['size_distribution']['large']}")
        
        if stats['oldest_file']:
            oldest_date = datetime.fromtimestamp(stats['oldest_file'][1]).strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"Oldest file: {Path(stats['oldest_file'][0]).name} ({oldest_date})")
        
        if stats['newest_file']:
            newest_date = datetime.fromtimestamp(stats['newest_file'][1]).strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"Newest file: {Path(stats['newest_file'][0]).name} ({newest_date})")
    else:
        print("📊 Directory Statistics:")
        print(f"Total files: {stats['total_files']:,}")
        print(f"Total size: {stats['total_size'] / (1024*1024*1024):.2f} GB")
        print(f"File types: {len(stats['file_types'])}")
        print("Size distribution:")
        print(f"  Small (<1MB): {stats['size_distribution']['small']}")
        print(f"  Medium (1-100MB): {stats['size_distribution']['medium']}")
        print(f"  Large (>100MB): {stats['size_distribution']['large']}")
        
        if stats['oldest_file']:
            oldest_date = datetime.fromtimestamp(stats['oldest_file'][1]).strftime("%Y-%m-%d %H:%M:%S")
            print(f"Oldest file: {Path(stats['oldest_file'][0]).name} ({oldest_date})")
        
        if stats['newest_file']:
            newest_date = datetime.fromtimestamp(stats['newest_file'][1]).strftime("%Y-%m-%d %H:%M:%S")
            print(f"Newest file: {Path(stats['newest_file'][0]).name} ({newest_date})")


class FileOrganizerScheduler:
    """Scheduler for automatic file organization"""
    
    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger or logging.getLogger("FileOrganizerScheduler")
        self.scheduler_running = False
        self.scheduled_directories: List[Dict] = []
    
    def add_scheduled_directory(self, directory: Path, frequency: str = "daily", time_str: str = "02:00", mode: str = "hybrid") -> bool:
        """
        Add directory to scheduled organization
        frequency: 'daily', 'weekly'
        time_str: '02:00' (24-hour format)
        mode: 'extension', 'date', 'size', 'hybrid'
        """
        if not SCHEDULE_AVAILABLE:
            self.logger.error("Schedule library not available. Install with: pip install schedule")
            return False
        
        if not directory.exists():
            self.logger.error(f"Cannot schedule non-existent directory: {directory}")
            return False
        
        schedule_info = {
            'directory': directory,
            'frequency': frequency,
            'time': time_str,
            'mode': mode,
            'last_run': None
        }
        
        self.scheduled_directories.append(schedule_info)
        
        # Schedule the job
        if frequency == "daily":
            schedule.every().day.at(time_str).do(self.scheduled_organize, directory, mode)
        elif frequency == "weekly":
            schedule.every().week.at(time_str).do(self.scheduled_organize, directory, mode)
        
        self.logger.info(f"📅 Scheduled {frequency} organization for {directory} at {time_str} (mode: {mode})")
        return True
    
    def scheduled_organize(self, directory: Path, mode: str = "hybrid"):
        """Function called by scheduler"""
        self.logger.info(f"🕒 Scheduled organization triggered for: {directory} (mode: {mode})")
        
        # Build extension mapping for hybrid mode
        extension_to_category = build_extension_to_category(DEFAULT_CATEGORY_EXTENSIONS.keys())
        
        if mode == "hybrid":
            organize_hybrid(directory, extension_to_category, self.logger)
        elif mode == "date":
            organize_by_date(directory, logger=self.logger)
        elif mode == "size":
            organize_by_size(directory, self.logger)
        else:  # extension mode
            # Use existing organize_files logic
            files = list(iter_files(directory, recursive=False, category_names=DEFAULT_CATEGORY_EXTENSIONS.keys(), logger=self.logger))
            plans = plan_moves(directory, files, extension_to_category, list(DEFAULT_CATEGORY_EXTENSIONS.keys()))
            execute_moves(plans, dry_run=False, logger=self.logger)
    
    def start_scheduler(self):
        """Start the scheduler in a background thread"""
        if not SCHEDULE_AVAILABLE:
            self.logger.error("Schedule library not available. Install with: pip install schedule")
            return False
        
        if self.scheduler_running:
            self.logger.info("Scheduler already running")
            return True
        
        def run_scheduler():
            self.scheduler_running = True
            self.logger.info("🕒 Scheduler started")
            
            while self.scheduler_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        return True
    
    def stop_scheduler(self):
        """Stop the scheduler"""
        self.scheduler_running = False
        if SCHEDULE_AVAILABLE:
            schedule.clear()
        self.logger.info("🕒 Scheduler stopped")


def create_test_files_advanced(base_directory: Optional[Path] = None) -> str:
    """Create advanced test files with various dates and sizes in a chosen folder.

    If base_directory is provided, the test set is created inside that directory
    under a subfolder named 'smart_test_directory'. Otherwise, it will be
    created in the current working directory.
    """
    base = Path(base_directory) if base_directory else Path.cwd()
    test_dir = base / "smart_test_directory"
    test_dir.mkdir(exist_ok=True)
    
    files_to_create = [
        # Documents with different dates
        ("old_document.pdf", "Old document content", 30),
        ("recent_report.docx", "Recent report content", 5),
        ("spreadsheet.xlsx", "X" * 1000, 2),
        
        # Images with different sizes  
        ("small_photo.jpg", "JPEG" + "x" * 500, 10),
        ("large_photo.png", "PNG" + "x" * 50000, 1),
        
        # Create duplicates
        ("duplicate1.txt", "Same content for testing", 7),
        ("duplicate2.txt", "Same content for testing", 7),
        ("duplicate3.txt", "Different content", 7),
        
        # Large files
        ("large_video.mp4", "VIDEO" + "x" * 2000000, 1),  # ~2MB
        ("huge_file.bin", "DATA" + "x" * 50000000, 1),    # ~50MB
    ]
    
    for filename, content, days_ago in files_to_create:
        file_path = test_dir / filename
        file_path.write_text(content)
        
        # Set different modification times
        past_time = time.time() - (days_ago * 24 * 3600)
        os.utime(file_path, (past_time, past_time))
    
    print(f"✓ Created advanced test files in: {test_dir}")
    return str(test_dir)


def interactive_mode(logger: Logger) -> None:
    """Interactive mode with menu options"""
    print("=" * 70)
    print("🤖 SMART FILE ORGANIZER - Phase 3")
    print("=" * 70)
    print("Smart Features:")
    print("🤖 Smart categorization (date/size/hybrid)")
    print("🔍 Duplicate detection with MD5 hashing")
    print("🕒 Automatic scheduler (daily/weekly)")
    print("📊 Advanced statistics and reporting")
    print("=" * 70)
    
    scheduler = FileOrganizerScheduler(logger)
    undo_manager = UndoRedoManager()
    
    while True:
        print("\n🚀 Smart Options:")
        print("1. Smart Organize (Extension + Date)")
        print("2. Organize by Date Only")
        print("3. Organize by Size Only") 
        print("4. Find & Remove Duplicates")
        print("5. Schedule Auto-Organization")
        print("6. Directory Statistics")
        print("7. Create Test Files")
        print("8. Start/Stop Scheduler")
        print("9. Undo last operation")
        print("10. Redo last operation")
        print("11. Open Enhanced GUI")
        print("12. Exit")
        
        try:
            choice = input("\nSelect option (1-12): ").strip()
            
            if choice == "1":
                directory = input("Enter directory path: ").strip().strip('"\'')
                if directory:
                    base = Path(directory)
                    extension_to_category = build_extension_to_category(DEFAULT_CATEGORY_EXTENSIONS.keys())
                    plans: List[MovePlan] = []
                    for item in base.iterdir():
                        if item.is_file():
                            created_date = datetime.fromtimestamp(item.stat().st_ctime)
                            date_folder = f"{created_date.year}/{created_date.strftime('%B')}"
                            category = determine_category(item, extension_to_category)
                            dest = base / category / date_folder / item.name
                            reason = "categorized"
                            if dest.exists():
                                dest = resolve_collision(dest)
                                reason = "name collision"
                            plans.append(MovePlan(source=item, destination=dest, reason=reason, size=item.stat().st_size, category=category))
                    executed: List[MovePlan] = []
                    for plan in plans:
                        try:
                            plan.destination.parent.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(plan.source), str(plan.destination))
                            executed.append(plan)
                        except Exception:
                            pass
                    if executed:
                        undo_manager.add_operation(UndoOperation("organize", datetime.now(), executed, f"Organized {len(executed)} files (hybrid)"))
                    print(f"✓ Smart organization complete! Moved {len(executed)} files")
            
            elif choice == "2":
                directory = input("Enter directory path: ").strip().strip('"\'')
                if directory:
                    base = Path(directory)
                    date_format = input("Date format (year_month/year_only/full_date): ").strip() or "year_month"
                    plans: List[MovePlan] = []
                    for item in base.iterdir():
                        if item.is_file():
                            created_date = datetime.fromtimestamp(item.stat().st_ctime)
                            if date_format == "year_month":
                                date_folder = f"{created_date.year}/{created_date.strftime('%B')}"
                            elif date_format == "year_only":
                                date_folder = str(created_date.year)
                            else:
                                date_folder = created_date.strftime("%Y/%m/%d")
                            dest = base / date_folder / item.name
                            reason = "categorized"
                            if dest.exists():
                                dest = resolve_collision(dest)
                                reason = "name collision"
                            plans.append(MovePlan(source=item, destination=dest, reason=reason, size=item.stat().st_size))
                    executed: List[MovePlan] = []
                    for plan in plans:
                        try:
                            plan.destination.parent.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(plan.source), str(plan.destination))
                            executed.append(plan)
                        except Exception:
                            pass
                    if executed:
                        undo_manager.add_operation(UndoOperation("organize", datetime.now(), executed, f"Organized {len(executed)} files (date)"))
                    print(f"✓ Date organization complete! Moved {len(executed)} files")
            
            elif choice == "3":
                directory = input("Enter directory path: ").strip().strip('"\'')
                if directory:
                    base = Path(directory)
                    plans: List[MovePlan] = []
                    for item in base.iterdir():
                        if item.is_file():
                            size = item.stat().st_size
                            if size < 1024 * 1024:
                                folder = "Small_Files"
                            elif size < 100 * 1024 * 1024:
                                folder = "Medium_Files"
                            else:
                                folder = "Large_Files"
                            dest = base / folder / item.name
                            reason = "categorized"
                            if dest.exists():
                                dest = resolve_collision(dest)
                                reason = "name collision"
                            plans.append(MovePlan(source=item, destination=dest, reason=reason, size=size))
                    executed: List[MovePlan] = []
                    for plan in plans:
                        try:
                            plan.destination.parent.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(plan.source), str(plan.destination))
                            executed.append(plan)
                        except Exception:
                            pass
                    if executed:
                        undo_manager.add_operation(UndoOperation("organize", datetime.now(), executed, f"Organized {len(executed)} files (size)"))
                    print(f"✓ Size organization complete! Moved {len(executed)} files")
            
            elif choice == "4":
                directory = input("Enter directory path: ").strip().strip('"\'')
                if directory:
                    remove_duplicates_interactive(Path(directory), logger)
            
            elif choice == "5":
                directory = input("Enter directory path to schedule: ").strip().strip('"\'')
                frequency = input("Frequency (daily/weekly): ").strip() or "daily"
                time_str = input("Time (HH:MM, 24-hour): ").strip() or "02:00"
                mode = input("Mode (extension/date/size/hybrid): ").strip() or "hybrid"
                
                if scheduler.add_scheduled_directory(Path(directory), frequency, time_str, mode):
                    print(f"✓ Scheduled {frequency} organization for {directory} at {time_str}")
                    if not scheduler.scheduler_running:
                        scheduler.start_scheduler()
            
            elif choice == "6":
                directory = input("Enter directory path: ").strip().strip('"\'')
                if directory:
                    print_directory_stats(Path(directory), logger)
            
            elif choice == "7":
                directory = input("Enter directory path to create test files (leave blank for current): ").strip().strip('\"\'')
                target = Path(directory) if directory else Path.cwd()
                test_dir = create_test_files_advanced(target)
                print(f"✓ Test files created in: {test_dir}")
            
            elif choice == "8":
                if scheduler.scheduler_running:
                    scheduler.stop_scheduler()
                    print("🕒 Scheduler stopped")
                else:
                    scheduler.start_scheduler()
                    print("🕒 Scheduler started")
            
            elif choice == "9":
                op = undo_manager.undo()
                if not op:
                    print("No operation to undo.")
                else:
                    undone = 0
                    for plan in op.plans:
                        try:
                            if plan.destination.exists():
                                shutil.move(str(plan.destination), str(plan.source))
                                undone += 1
                        except Exception as e:
                            print(f"Error undoing {plan.destination}: {e}")
                    print(f"✓ Undo complete: {undone} files restored")

            elif choice == "10":
                op = undo_manager.redo()
                if not op:
                    print("No operation to redo.")
                else:
                    redone = 0
                    for plan in op.plans:
                        try:
                            if plan.source.exists():
                                plan.destination.parent.mkdir(parents=True, exist_ok=True)
                                shutil.move(str(plan.source), str(plan.destination))
                                redone += 1
                        except Exception as e:
                            print(f"Error redoing {plan.source}: {e}")
                    print(f"✓ Redo complete: {redone} files moved")

            elif choice == "12":
                scheduler.stop_scheduler()
                print("👋 Goodbye!")
                break
            
            else:
                # Optionally launch GUI via 12
                if choice == "11":
                    try:
                        print("🚀 Launching Enhanced Smart File Organizer GUI...")
                        app = EnhancedFileOrganizerGUI(logger)
                        app.run()
                    except Exception as e:
                        print(f"❌ Unable to launch GUI: {e}")
                else:
                    print("❌ Invalid choice. Please select 1-12.")
                
        except KeyboardInterrupt:
            print("\n\n⚠️ Operation cancelled by user.")
            scheduler.stop_scheduler()
            break
        except Exception as e:
            print(f"❌ Error: {e}")


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
    # Duplicate detection arguments
    parser.add_argument(
        "--find-duplicates",
        action="store_true",
        help="Find and report duplicate files using MD5 hashing",
    )
    parser.add_argument(
        "--remove-duplicates",
        action="store_true",
        help="Interactive duplicate removal (requires --find-duplicates)",
    )
    # Organization mode arguments
    parser.add_argument(
        "--mode",
        type=str,
        default="extension",
        choices=["extension", "date", "size", "hybrid"],
        help="Organization mode: extension (default), date, size, or hybrid",
    )
    parser.add_argument(
        "--date-format",
        type=str,
        default="year_month",
        choices=["year_month", "year_only", "full_date"],
        help="Date format for date organization (default: year_month)",
    )
    # Statistics argument
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show comprehensive directory statistics",
    )
    # Scheduler arguments
    parser.add_argument(
        "--schedule",
        type=str,
        choices=["daily", "weekly"],
        help="Schedule automatic organization (daily or weekly)",
    )
    parser.add_argument(
        "--time",
        type=str,
        default="02:00",
        help="Time for scheduled organization (HH:MM format, default: 02:00)",
    )
    # Interactive mode argument
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode with menu options",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Launch CLI menu (alias of --interactive)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch Enhanced GUI",
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
        # If no actionable flags provided, prefer GUI when a TTY is not required
        if not (args.interactive or args.cli or args.find_duplicates or args.remove_duplicates or args.stats or args.schedule) and args.mode == "extension" and not args.dry_run and args.gui:
            # explicit --gui handled above; this check keeps logic readable for future default switch
            pass
        # GUI mode
        if args.gui:
            try:
                print("🚀 Launching Enhanced Smart File Organizer...")
                app = EnhancedFileOrganizerGUI(logger)
                app.run()
            except Exception as exc:  # noqa: BLE001
                print(f"❌ Enhanced GUI failed to start: {exc}")
                print("📝 Note: For drag & drop support, install: pip install tkinterdnd2")
                interactive_mode(logger)
            return
        # Interactive mode
        if args.interactive or args.cli:
            interactive_mode(logger)
            return
        
        base = Path(args.path).expanduser().resolve()

        if not base.exists() or not base.is_dir():
            logger.error("Path does not exist or is not a directory: %s", base)
            raise SystemExit(2)

        enabled_categories = [c.strip() for c in args.categories.split(",") if c.strip()]
        extension_to_category = build_extension_to_category(enabled_categories)

        # Handle statistics
        if args.stats:
            print_directory_stats(base, logger)
            return

        # Handle scheduling
        if args.schedule:
            scheduler = FileOrganizerScheduler(logger)
            if scheduler.add_scheduled_directory(base, args.schedule, args.time, args.mode):
                logger.info(f"✓ Scheduled {args.schedule} organization for {base} at {args.time}")
                scheduler.start_scheduler()
                logger.info("🕒 Scheduler started. Press Ctrl+C to stop.")
                try:
                    while True:
                        time.sleep(60)
                except KeyboardInterrupt:
                    scheduler.stop_scheduler()
            return

        # Handle duplicate detection
        if args.find_duplicates or args.remove_duplicates:
            duplicates = find_duplicates(base, logger)
            if args.remove_duplicates:
                remove_duplicates_interactive(base, logger)
            return

        # Handle date organization mode
        if args.mode == "date":
            if args.dry_run:
                logger.info("DRY-RUN: Would organize files by date")
                # In dry-run mode, just show what would happen
                file_count = sum(1 for f in base.iterdir() if f.is_file())
                logger.info(f"Would organize {file_count} files by date using format: {args.date_format}")
            else:
                moved = organize_by_date(base, args.date_format, logger)
                logger.info("Date organization complete. Moved: %d files", moved)
            return

        # Handle size organization mode
        if args.mode == "size":
            if args.dry_run:
                logger.info("DRY-RUN: Would organize files by size")
                # In dry-run mode, just show what would happen
                file_count = sum(1 for f in base.iterdir() if f.is_file())
                logger.info(f"Would organize {file_count} files by size")
            else:
                moved = organize_by_size(base, logger)
                logger.info("Size organization complete. Moved: %d files", moved)
            return

        # Handle hybrid organization mode
        if args.mode == "hybrid":
            if args.dry_run:
                logger.info("DRY-RUN: Would organize files by hybrid (type + date)")
                # In dry-run mode, just show what would happen
                file_count = sum(1 for f in base.iterdir() if f.is_file())
                logger.info(f"Would organize {file_count} files by hybrid mode")
            else:
                moved = organize_hybrid(base, extension_to_category, logger)
                logger.info("Hybrid organization complete. Moved: %d files", moved)
            return

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


class EnhancedFileOrganizerGUI:
    """Complete GUI to match the target layout - ensuring both panels are visible"""
    
    def __init__(self, logger: Logger):
        self.logger = logger
        self.org = EnhancedSmartFileOrganizer(logger)
        self.progress_tracker = ProgressTracker(callback=self._on_progress_update)
        self.undo_manager = UndoRedoManager()
        
        # Initialize root window with drag-and-drop support if available
        if DND_AVAILABLE:
            self.root = TkinterDnD.Tk()
            self._dnd_enabled = True
        else:
            self.root = tk.Tk()
            self._dnd_enabled = False
        
        self.root.title("Smart File Organizer - Enhanced")
        self.root.geometry("1400x900")  # Wider window to accommodate both panels
        self.root.minsize(1200, 700)
        
        # Configure modern styling
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors through style system
        style.configure('Title.TLabel', font=('Segoe UI', 16, 'bold'), foreground='#2c3e50')
        style.configure('Header.TLabel', font=('Segoe UI', 12, 'bold'), foreground='#34495e')
        style.configure('Custom.TButton', font=('Segoe UI', 10), padding=(6, 3))
        style.configure('Action.TButton', font=('Segoe UI', 11, 'bold'), padding=(10, 5))
        style.configure('Treeview', rowheight=22)
        style.map('Action.TButton', background=[('active', '#3498db')])
        
        # Variables
        self.selected_folder = tk.StringVar()
        self.mode = tk.StringVar(value="extension")
        self.date_format = tk.StringVar(value="year_month")
        self.preview_mode = tk.BooleanVar(value=True)
        
        # Progress variables
        self.progress_var = tk.DoubleVar(value=0.0)
        self.status_var = tk.StringVar(value="Ready")
        self.current_file_var = tk.StringVar(value="")
        self.stats_var = tk.StringVar(value="")
        self._last_progress_log = -10
        
        # Preview data
        self.current_plans: List[MovePlan] = []
        self._working = False
        
        self._build_ui()
        self._update_undo_buttons()
        self.org.set_callbacks(progress_callback=self._on_progress, status_callback=self._on_status)
    
    def _build_ui(self):
        # Main container with padding
        main_container = ttk.Frame(self.root, padding=8)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Title section
        self._build_title_section(main_container)
        
        # CRITICAL: Two-column layout using grid to avoid unintended gaps
        columns_frame = ttk.Frame(main_container)
        columns_frame.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        columns_frame.columnconfigure(0, weight=0)
        columns_frame.columnconfigure(1, weight=1)  # Right panel expands
        columns_frame.rowconfigure(0, weight=1)

        # LEFT PANEL - Size to content (no fixed width)
        left_panel_container = ttk.Frame(columns_frame)
        left_panel_container.grid(row=0, column=0, sticky=tk.NS, padx=(0, 5))
        try:
            left_panel_container.configure(width=320)  # compact fixed width to remove extra spacing
            left_panel_container.grid_propagate(False)
        except Exception:
            pass

        # RIGHT PANEL - Takes remaining space
        right_panel_container = ttk.Frame(columns_frame)
        right_panel_container.grid(row=0, column=1, sticky=tk.NSEW, padx=(5, 0))
        
        self._build_left_panel(left_panel_container)
        self._build_right_panel(right_panel_container)
    
    def _build_title_section(self, parent):
        title_frame = ttk.Frame(parent)
        title_frame.pack(fill=tk.X, pady=(0, 6))
        
        # Title with icon
        title_container = ttk.Frame(title_frame)
        title_container.pack(side=tk.LEFT)
        
        title_label = ttk.Label(title_container, text="📁 Smart File Organizer", style='Title.TLabel')
        title_label.pack()
        
        subtitle_label = tk.Label(title_container, text="Organize your files intelligently", 
                                font=('Segoe UI', 9), foreground='#041D1F', bg='#D6D8D8')
        subtitle_label.pack()
        
        # Undo/Redo buttons on the right
        undo_frame = ttk.Frame(title_frame)
        undo_frame.pack(side=tk.RIGHT)
        
        self.undo_button = ttk.Button(undo_frame, text="⟲ Undo", command=self._undo, 
                                    state=tk.DISABLED, style='Custom.TButton')
        self.undo_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.redo_button = ttk.Button(undo_frame, text="⟳ Redo", command=self._redo, 
                                    state=tk.DISABLED, style='Custom.TButton')
        self.redo_button.pack(side=tk.LEFT)
    
    def _build_left_panel(self, parent):
        # Split left panel into scrollable top content area and fixed bottom actions
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        # Scrollable content area
        canvas = tk.Canvas(parent, highlightthickness=0)
        vscroll = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=vscroll.set)

        canvas.grid(row=0, column=0, sticky=tk.NSEW)
        vscroll.grid(row=0, column=1, sticky=tk.NS)

        # Mouse wheel binding for smooth scroll
        def _on_mousewheel(event):
            delta = -1 * (event.delta // 120) if event.delta else (1 if event.num == 5 else -1)
            canvas.yview_scroll(delta, "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", _on_mousewheel)  # Linux up
        canvas.bind_all("<Button-5>", _on_mousewheel)  # Linux down
        
        # LEFT PANEL CONTENT
        # Folder Selection Section
        folder_section = ttk.LabelFrame(scrollable_frame, text="Select Folder", padding=8)
        folder_section.pack(fill=tk.X, pady=(0, 6))
        
        folder_entry_frame = ttk.Frame(folder_section)
        folder_entry_frame.pack(fill=tk.X)
        
        self.folder_entry = ttk.Entry(folder_entry_frame, textvariable=self.selected_folder)
        self.folder_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        
        browse_button = ttk.Button(folder_entry_frame, text="Browse", 
                                 command=self._browse_folder, style='Custom.TButton')
        browse_button.pack(side=tk.RIGHT)
        
        # Enable drag-and-drop if available
        if self._dnd_enabled:
            self.folder_entry.drop_target_register(DND_FILES)
            self.folder_entry.dnd_bind('<<Drop>>', self._on_folder_drop)
            
            hint_label = tk.Label(folder_section, text="💡 Tip: You can drag and drop a folder here!", 
                                font=('Segoe UI', 8), foreground='#041D1F', bg='#D6D8D8')
            hint_label.pack(pady=(8, 0))
        
        # Organization Options Section
        options_section = ttk.LabelFrame(scrollable_frame, text="Organization Options", padding=8)
        options_section.pack(fill=tk.X, pady=(0, 6))
        
        # Mode selection
        ttk.Label(options_section, text="Organization Mode:", style='Header.TLabel').pack(anchor=tk.W, pady=(0, 8))
        
        modes_frame = ttk.Frame(options_section)
        modes_frame.pack(fill=tk.X, pady=(0, 6))
        
        modes = [
            ("📄 Extension", "extension", "Group by file type"),
            ("📅 Date", "date", "Group by creation date"), 
            ("📏 Size", "size", "Group by file size"),
            ("🔄 Hybrid", "hybrid", "Type + Date combined")
        ]
        
        for i, (text, value, desc) in enumerate(modes):
            mode_frame = ttk.Frame(modes_frame, relief='solid', borderwidth=1, padding=4)
            mode_frame.pack(fill=tk.X, pady=0)
            
            rb = ttk.Radiobutton(mode_frame, text=text, value=value, variable=self.mode,
                               command=self._on_mode_change)
            rb.pack(anchor=tk.W)
            
            desc_label = tk.Label(mode_frame, text=desc, font=('Segoe UI', 8), 
                                foreground='#041D1F', bg='#D6D8D8')
            desc_label.pack(anchor=tk.W, padx=(18, 0))
        
        # Date format options (initially hidden)
        self.date_options_frame = ttk.LabelFrame(options_section, text="Date Format", padding=6)
        
        date_formats = [
            ("📁 Year/Month", "year_month"),
            ("📁 Year Only", "year_only"),
            ("📁 Full Date", "full_date")
        ]
        
        for text, value in date_formats:
            rb = ttk.Radiobutton(self.date_options_frame, text=text, value=value, 
                               variable=self.date_format)
            rb.pack(anchor=tk.W, pady=1)
        
        # Preview mode checkbox
        preview_frame = ttk.Frame(options_section)
        preview_frame.pack(fill=tk.X, pady=(6, 0))
        
        self.preview_check = ttk.Checkbutton(preview_frame, 
                                      text="🔍 Preview mode (show changes before applying)", 
                                      variable=self.preview_mode)
        self.preview_check.pack(anchor=tk.W)
        
        # Spacer pushes actions to bottom so they remain visible on resize
        # ACTIONS SECTION (pinned to bottom of left panel)
        actions_section = ttk.LabelFrame(parent, text="Actions", padding=8)
        actions_section.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=(0, 6))
        
        # Main action button
        self.start_button = ttk.Button(actions_section, text="🚀 Start Organization", 
                                     command=self._start_organization, style='Action.TButton')
        self.start_button.pack(fill=tk.X, pady=(0, 6))
        
        # Secondary buttons
        secondary_frame = ttk.Frame(actions_section)
        secondary_frame.pack(fill=tk.X, pady=(0, 6))
        
        # First row
        button_row1 = ttk.Frame(secondary_frame)
        button_row1.pack(fill=tk.X, pady=(0, 4))
        
        self.clean_button = ttk.Button(button_row1, text="🧹 Clean Preview", 
                                     command=self._clear_preview, style='Custom.TButton')
        self.clean_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        
        self.duplicates_button = ttk.Button(button_row1, text="🔍 Find Duplicates", 
                                          command=self._find_duplicates, style='Custom.TButton')
        self.duplicates_button.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))
        
        # Second row
        self.stats_button = ttk.Button(secondary_frame, text="📊 Show Statistics", 
                                     command=self._show_stats, style='Custom.TButton')
        self.stats_button.pack(fill=tk.X)
        
        # Progress section removed — progress will be logged in the right panel
    
    def _build_right_panel(self, parent):
        # Preview Section
        preview_section = ttk.LabelFrame(parent, text="Preview Changes", padding=8)
        preview_section.pack(fill=tk.BOTH, expand=True)
        
        # Preview controls
        preview_controls = ttk.Frame(preview_section)
        preview_controls.pack(fill=tk.X, pady=(0, 6))
        
        ttk.Label(preview_controls, text="📋 Planned Operations", 
                 style='Header.TLabel').pack(side=tk.LEFT)
        
        # Apply/Cancel buttons
        preview_buttons = ttk.Frame(preview_controls)
        preview_buttons.pack(side=tk.RIGHT)
        
        self.apply_button = ttk.Button(preview_buttons, text="✅ Apply Changes", 
                                     command=self._apply_preview, state=tk.DISABLED,
                                     style='Action.TButton')
        self.apply_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.cancel_button = ttk.Button(preview_buttons, text="❌ Cancel", 
                                      command=self._clear_preview, style='Custom.TButton')
        self.cancel_button.pack(side=tk.LEFT)
        
        # Preview tree with scrollbars
        tree_container = ttk.Frame(preview_section)
        tree_container.pack(fill=tk.BOTH, expand=True)
        
        # Create treeview with columns
        columns = ('source', 'destination', 'category', 'size', 'reason')
        self.preview_tree = ttk.Treeview(tree_container, columns=columns, show='headings')
        
        # Define column headings
        self.preview_tree.heading('source', text='📄 Source File')
        self.preview_tree.heading('destination', text='📁 Destination')
        self.preview_tree.heading('category', text='🏷️ Category')
        self.preview_tree.heading('size', text='📏 Size')
        self.preview_tree.heading('reason', text='ℹ️ Reason')
        
        self.preview_tree.column('source', width=200)
        self.preview_tree.column('destination', width=250)
        self.preview_tree.column('category', width=100)
        self.preview_tree.column('size', width=80)
        self.preview_tree.column('reason', width=120)
        
        # Add scrollbars
        tree_scroll_v = ttk.Scrollbar(tree_container, orient=tk.VERTICAL, command=self.preview_tree.yview)
        tree_scroll_h = ttk.Scrollbar(tree_container, orient=tk.HORIZONTAL, command=self.preview_tree.xview)
        
        self.preview_tree.configure(yscrollcommand=tree_scroll_v.set, xscrollcommand=tree_scroll_h.set)
        
        # Pack scrollbars and treeview
        tree_scroll_v.pack(side=tk.RIGHT, fill=tk.Y)
        tree_scroll_h.pack(side=tk.BOTTOM, fill=tk.X)
        self.preview_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Summary section
        summary_section = ttk.LabelFrame(parent, text="Operation Summary", padding=8)
        summary_section.pack(fill=tk.X, pady=(6, 0))
        
        self.summary_text = tk.Text(summary_section, height=6, font=('Segoe UI', 9),
                                  wrap=tk.WORD, relief='flat', bg='#f8f9fa')
        self.summary_text.pack(fill=tk.BOTH, expand=True)

    def _on_mode_change(self):
        """Handle mode change and show/hide date format options"""
        if self.mode.get() == "date":
            self.date_options_frame.pack(fill=tk.X, pady=(8, 0))
        else:
            self.date_options_frame.pack_forget()
    
    def _browse_folder(self):
        """Open folder selection dialog"""
        folder = filedialog.askdirectory(title="Select folder to organize")
        if folder:
            self.selected_folder.set(folder)
    
    def _on_folder_drop(self, event):
        """Handle dropped folder paths"""
        data = str(event.data).strip()
        if not data:
            return
        
        # Parse dropped paths
        paths = []
        token = ''
        in_brace = False
        
        for ch in data:
            if ch == '{':
                in_brace = True
                token = ''
            elif ch == '}':
                in_brace = False
                if token:
                    paths.append(token)
                token = ''
            elif ch == ' ' and not in_brace:
                if token:
                    paths.append(token)
                token = ''
            else:
                token += ch
        
        if token:
            paths.append(token)
        
        if paths and os.path.isdir(paths[0]):
            self.selected_folder.set(paths[0])
            self._log(f"📁 Folder dropped: {paths[0]}")
    
    def _validate_folder(self) -> Optional[Path]:
        """Validate selected folder"""
        folder_path = self.selected_folder.get().strip()
        if not folder_path:
            messagebox.showerror("Error", "Please select a folder")
            return None
        
        path = Path(folder_path)
        if not path.exists() or not path.is_dir():
            messagebox.showerror("Error", "Invalid folder path")
            return None
        
        return path
    
    def _log(self, message: str):
        """Add timestamped message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        
        # Update summary text
        current = self.summary_text.get(1.0, tk.END).strip()
        if current:
            self.summary_text.insert(tk.END, "\n" + full_message)
        else:
            self.summary_text.insert(tk.END, full_message)
        
        self.summary_text.see(tk.END)
        self.root.update_idletasks()

    def _on_progress(self, value: float) -> None:
        """Progress callback from organizer — also log periodically"""
        self.progress_var.set(value)
        try:
            step = int(value // 10) * 10
            if step >= 0 and step != self._last_progress_log and step % 10 == 0:
                self._last_progress_log = step
                self._log(f"Progress: {step}%")
        except Exception:
            pass
        self.root.update_idletasks()

    def _on_status(self, msg: str) -> None:
        """Status callback from organizer — also mirror to summary log"""
        self.status_var.set(msg)
        try:
            self._log(msg)
        except Exception:
            pass
        self.root.update_idletasks()

    def _on_progress_update(self, progress_info: Dict[str, Any]):
        """Enhanced progress update with detailed statistics"""
        self.progress_var.set(progress_info['progress_percent'])
        
        # Update current file (truncate if too long)
        current_file = progress_info['current_file']
        if len(current_file) > 30:
            current_file = "..." + current_file[-27:]
        self.current_file_var.set(current_file)
        
        # Update statistics
        stats = []
        if progress_info['total_files'] > 0:
            stats.append(f"{progress_info['processed_files']}/{progress_info['total_files']}")
        
        if progress_info['total_size'] > 0:
            processed_mb = progress_info['processed_size'] / (1024 * 1024)
            total_mb = progress_info['total_size'] / (1024 * 1024)
            stats.append(f"{processed_mb:.1f}/{total_mb:.1f}MB")
        
        if progress_info['eta'] > 0:
            eta_min = int(progress_info['eta'] / 60)
            eta_sec = int(progress_info['eta'] % 60)
            stats.append(f"{eta_min:02d}:{eta_sec:02d}")
        
        if progress_info['errors'] > 0:
            stats.append(f"❌{progress_info['errors']}")
        
        self.stats_var.set(" | ".join(stats))
        self.root.update_idletasks()

    def _start_organization(self):
        """Start the organization process"""
        if self._working:
            messagebox.showwarning("Busy", "An operation is already running")
            return
        
        folder_path = self._validate_folder()
        if not folder_path:
            return
        
        def work():
            try:
                self._working = True
                self.start_button.config(state=tk.DISABLED)
                
                # Generate plans
                self._log("🔍 Scanning files...")
                self.status_var.set("Scanning...")
                
                plans = self._generate_plans(folder_path)
                if not plans:
                    self._log("ℹ️ No files to organize")
                    self.status_var.set("No files found")
                    return
                
                # Store plans for preview
                self.current_plans = plans
                
                if self.preview_mode.get():
                    # Show preview
                    self._show_preview()
                    self._log(f"📋 Preview generated: {len(plans)} planned moves")
                    self.status_var.set("Preview ready")
                else:
                    # Execute immediately
                    self._execute_plans(plans)
                
            except Exception as e:
                self._log(f"❌ Error: {str(e)}")
                messagebox.showerror("Error", str(e))
            finally:
                self._working = False
                self.start_button.config(state=tk.NORMAL)
                self.progress_var.set(0)
                self.current_file_var.set("")
                self.stats_var.set("")
        
        threading.Thread(target=work, daemon=True).start()

    def _generate_plans(self, folder_path: Path) -> List[MovePlan]:
        """Generate organization plans based on selected mode"""
        extension_to_category = self.org.build_extension_to_category(DEFAULT_CATEGORY_EXTENSIONS.keys())
        plans = []
        
        # Get all files
        files = []
        for item in folder_path.iterdir():
            if item.is_file():
                files.append(item)
        
        self.progress_tracker.set_total(len(files))
        
        for i, file_path in enumerate(files):
            try:
                file_size = file_path.stat().st_size
                category = self.org.determine_category(file_path, extension_to_category)
                
                # Generate destination based on mode
                if self.mode.get() == "date":
                    destination = self._get_date_destination(folder_path, file_path)
                elif self.mode.get() == "size":
                    destination = self._get_size_destination(folder_path, file_path)
                elif self.mode.get() == "hybrid":
                    destination = self._get_hybrid_destination(folder_path, file_path, category)
                else:  # extension mode
                    destination = folder_path / category / file_path.name
                
                # Handle name collisions
                if destination.exists():
                    destination = self.org.resolve_collision(destination)
                    reason = "name collision"
                else:
                    reason = "categorized"
                
                plan = MovePlan(
                    source=file_path,
                    destination=destination,
                    reason=reason,
                    size=file_size,
                    category=category
                )
                plans.append(plan)
                
                self.progress_tracker.update(
                    current_file=file_path.name,
                    file_size=file_size,
                    category=category
                )
                
            except Exception as e:
                self._log(f"❌ Error processing {file_path}: {e}")
                self.progress_tracker.update(error=True)
        
        return plans

    def _get_date_destination(self, base_path: Path, file_path: Path) -> Path:
        """Get destination path for date-based organization"""
        stat = file_path.stat()
        created_date = datetime.fromtimestamp(stat.st_ctime)
        
        if self.date_format.get() == "year_month":
            date_folder = f"{created_date.year}/{created_date.strftime('%B')}"
        elif self.date_format.get() == "year_only":
            date_folder = str(created_date.year)
        else:  # full_date
            date_folder = created_date.strftime("%Y/%m/%d")
        
        return base_path / date_folder / file_path.name

    def _get_size_destination(self, base_path: Path, file_path: Path) -> Path:
        """Get destination path for size-based organization"""
        size = file_path.stat().st_size
        
        if size < 1024 * 1024:  # < 1MB
            size_folder = "Small_Files"
        elif size < 100 * 1024 * 1024:  # < 100MB
            size_folder = "Medium_Files"
        else:
            size_folder = "Large_Files"
        
        return base_path / size_folder / file_path.name

    def _get_hybrid_destination(self, base_path: Path, file_path: Path, category: str) -> Path:
        """Get destination path for hybrid organization"""
        stat = file_path.stat()
        created_date = datetime.fromtimestamp(stat.st_ctime)
        date_folder = f"{created_date.year}/{created_date.strftime('%B')}"
        
        return base_path / category / date_folder / file_path.name

    def _show_preview(self):
        """Show preview of planned changes"""
        # Clear existing items
        for item in self.preview_tree.get_children():
            self.preview_tree.delete(item)
        
        # Add plans to tree
        category_counts = Counter()
        total_size = 0
        
        for plan in self.current_plans:
            # Format size
            if plan.size < 1024:
                size_str = f"{plan.size}B"
            elif plan.size < 1024 * 1024:
                size_str = f"{plan.size / 1024:.1f}KB"
            else:
                size_str = f"{plan.size / (1024 * 1024):.1f}MB"
            
            # Get relative destination path for display
            try:
                rel_dest = plan.destination.relative_to(plan.source.parent)
                dest_display = str(rel_dest)
            except ValueError:
                dest_display = str(plan.destination)
            
            # Add to tree
            self.preview_tree.insert('', 'end', values=(
                plan.source.name,
                dest_display,
                plan.category,
                size_str,
                plan.reason
            ))
            
            category_counts[plan.category] += 1
            total_size += plan.size
        
        # Enable apply button
        self.apply_button.config(state=tk.NORMAL)
        
        # Update summary
        summary_text = f"📊 Organization Summary\n"
        summary_text += f"{'='*40}\n\n"
        summary_text += f"📁 Files to organize: {len(self.current_plans)}\n"
        summary_text += f"💾 Total size: {total_size / (1024 * 1024):.2f} MB\n"
        summary_text += f"🎯 Mode: {self.mode.get().title()}\n\n"
        summary_text += f"📂 Categories:\n"
        for cat, count in sorted(category_counts.items()):
            summary_text += f"   • {cat}: {count} files\n"
        
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(1.0, summary_text)
    
    def _apply_preview(self):
        """Apply the previewed changes"""
        if not self.current_plans:
            return
        
        def work():
            try:
                self._working = True
                self.apply_button.config(state=tk.DISABLED)
                self._execute_plans(self.current_plans)
            finally:
                self._working = False
                self.apply_button.config(state=tk.NORMAL)
        
        threading.Thread(target=work, daemon=True).start()
    
    def _execute_plans(self, plans: List[MovePlan]):
        """Execute the organization plans"""
        self._log(f"⚡ Executing {len(plans)} file operations...")
        self.status_var.set("Executing...")
        
        executed_plans = []
        self.progress_tracker.reset()
        self.progress_tracker.set_total(len(plans), sum(p.size for p in plans))
        
        for plan in plans:
            try:
                # Create destination directory
                plan.destination.parent.mkdir(parents=True, exist_ok=True)
                
                # Move file
                shutil.move(str(plan.source), str(plan.destination))
                executed_plans.append(plan)
                
                self.progress_tracker.update(
                    current_file=plan.source.name,
                    file_size=plan.size,
                    category=plan.category
                )
                
                self._log(f"📁 Moved: {plan.source.name} → {plan.destination.parent.name}/")
                
            except Exception as e:
                self._log(f"❌ Error moving {plan.source.name}: {e}")
                self.progress_tracker.update(error=True)
        
        # Add to undo history
        if executed_plans:
            operation = UndoOperation(
                operation_type="organize",
                timestamp=datetime.now(),
                plans=executed_plans,
                description=f"Organized {len(executed_plans)} files ({self.mode.get()} mode)"
            )
            self.undo_manager.add_operation(operation)
            self._update_undo_buttons()
        
        self._log(f"✅ Complete: {len(executed_plans)} moved, {self.progress_tracker.errors} errors")
        self.status_var.set("Complete")
        self._clear_preview()
    
    def _clear_preview(self):
        """Clear the preview display"""
        self.current_plans = []
        for item in self.preview_tree.get_children():
            self.preview_tree.delete(item)
        self.summary_text.delete(1.0, tk.END)
        self.apply_button.config(state=tk.DISABLED)
    
    def _undo(self):
        """Undo the last operation"""
        operation = self.undo_manager.undo()
        if not operation:
            return
        
        def work():
            try:
                self._working = True
                self._log(f"⟲ Undoing: {operation.description}")
                self.status_var.set("Undoing...")
                
                undone_count = 0
                for plan in operation.plans:
                    try:
                        if plan.destination.exists():
                            shutil.move(str(plan.destination), str(plan.source))
                            undone_count += 1
                            self._log(f"🔄 Restored: {plan.source.name}")
                    except Exception as e:
                        self._log(f"❌ Error undoing {plan.destination.name}: {e}")
                
                self._log(f"✅ Undo complete: {undone_count} files restored")
                self.status_var.set("Undo complete")
                
            finally:
                self._working = False
                self._update_undo_buttons()
        
        threading.Thread(target=work, daemon=True).start()
    
    def _redo(self):
        """Redo the previously undone operation"""
        operation = self.undo_manager.redo()
        if not operation:
            return
        
        def work():
            try:
                self._working = True
                self._log(f"⟳ Redoing: {operation.description}")
                self.status_var.set("Redoing...")
                
                redone_count = 0
                for plan in operation.plans:
                    try:
                        if plan.source.exists():
                            plan.destination.parent.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(plan.source), str(plan.destination))
                            redone_count += 1
                            self._log(f"📁 Moved: {plan.source.name}")
                    except Exception as e:
                        self._log(f"❌ Error redoing {plan.source.name}: {e}")
                
                self._log(f"✅ Redo complete: {redone_count} files moved")
                self.status_var.set("Redo complete")
                
            finally:
                self._working = False
                self._update_undo_buttons()
        
        threading.Thread(target=work, daemon=True).start()
    
    def _update_undo_buttons(self):
        """Update undo/redo button states"""
        if self.undo_manager.can_undo():
            self.undo_button.config(state=tk.NORMAL)
            desc = self.undo_manager.operations[self.undo_manager.current_index].description
            self.undo_button.config(text=f"⟲ Undo ({desc[:10]}...)")
        else:
            self.undo_button.config(state=tk.DISABLED, text="⟲ Undo")
        
        if self.undo_manager.can_redo():
            self.redo_button.config(state=tk.NORMAL)
            desc = self.undo_manager.operations[self.undo_manager.current_index + 1].description
            self.redo_button.config(text=f"⟳ Redo ({desc[:10]}...)")
        else:
            self.redo_button.config(state=tk.DISABLED, text="⟳ Redo")
    
    def _find_duplicates(self):
        """Find duplicate files"""
        folder_path = self._validate_folder()
        if not folder_path:
            return
        
        
        def work():
            try:
                self._working = True
                self.status_var.set("Finding duplicates...")
                self._log("🔍 Scanning for duplicate files...")
                
                duplicates = self.org.find_duplicates_in(folder_path)
                
                if not duplicates:
                    self._log("✅ No duplicate files found")
                    messagebox.showinfo("Duplicates", "No duplicate files found!")
                    return
                
                # Show duplicates in summary
                duplicate_text = f"🔍 Duplicate Files Found\n"
                duplicate_text += f"{'='*40}\n\n"
                duplicate_text += f"Found {len(duplicates)} duplicate groups:\n\n"
                
                for i, group in enumerate(duplicates[:8], 1):
                    duplicate_text += f"Group {i}:\n"
                    for file_path in group:
                        size_mb = file_path.stat().st_size / (1024 * 1024)
                        duplicate_text += f"  • {file_path.name} ({size_mb:.2f}MB)\n"
                        duplicate_text += f"    {file_path.parent}\n"
                    duplicate_text += "\n"
                
                if len(duplicates) > 8:
                    duplicate_text += f"... and {len(duplicates) - 8} more groups\n"
                
                self.summary_text.delete(1.0, tk.END)
                self.summary_text.insert(1.0, duplicate_text)
                
                self._log(f"✅ Found {len(duplicates)} duplicate groups")
                
            except Exception as e:
                self._log(f"❌ Error finding duplicates: {e}")
                messagebox.showerror("Error", str(e))
            finally:
                self._working = False
                self.status_var.set("Ready")
        
        threading.Thread(target=work, daemon=True).start()
    
    def _show_stats(self):
        """Show directory statistics"""
        folder_path = self._validate_folder()
        if not folder_path:
            return
        
        def work():
            try:
                self._working = True
                self.status_var.set("Calculating statistics...")
                self._log("📊 Calculating directory statistics...")
                
                stats = self.org.get_stats(folder_path)
                
                # Format statistics for display
                stats_text = f"📊 Directory Statistics\n"
                stats_text += f"{'='*40}\n\n"
                stats_text += f"📁 Total files: {stats['total_files']:,}\n"
                stats_text += f"💾 Total size: {stats['total_size'] / (1024*1024*1024):.2f} GB\n\n"
                
                stats_text += f"📏 Size Distribution:\n"
                stats_text += f"  • Small (<1MB): {stats['size_distribution']['small']:,}\n"
                stats_text += f"  • Medium (1-100MB): {stats['size_distribution']['medium']:,}\n"
                stats_text += f"  • Large (>100MB): {stats['size_distribution']['large']:,}\n\n"
                
                stats_text += f"📂 Top Categories:\n"
                for category, count in stats['categories'].most_common(5):
                    percentage = (count / stats['total_files']) * 100
                    stats_text += f"  • {category}: {count:,} ({percentage:.1f}%)\n"
                
                stats_text += f"\n🔤 Top File Types:\n"
                for ext, count in stats['file_types'].most_common(5):
                    ext_display = ext if ext else "(no ext)"
                    percentage = (count / stats['total_files']) * 100
                    stats_text += f"  • {ext_display}: {count:,} ({percentage:.1f}%)\n"
                
                self.summary_text.delete(1.0, tk.END)
                self.summary_text.insert(1.0, stats_text)
                
                self._log("✅ Statistics calculated successfully")
                
            except Exception as e:
                self._log(f"❌ Error calculating stats: {e}")
                messagebox.showerror("Error", str(e))
            finally:
                self._working = False
                self.status_var.set("Ready")
        
        threading.Thread(target=work, daemon=True).start()
    
    def run(self):
        """Start the GUI application"""
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.root.mainloop()
    
    def _on_closing(self):
        """Handle application closing"""
        if self._working:
            if messagebox.askokcancel("Quit", "An operation is running. Do you want to quit anyway?"):
                self.root.destroy()
        else:
            self.root.destroy()


if __name__ == "__main__":
    main()
