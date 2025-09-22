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
import hashlib
import os
import shutil
import time
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import logging
from logging import Logger
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False


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


@dataclass
class DuplicateGroup:
    files: List[Path]
    hash_value: str
    total_size: int


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


def create_test_files_advanced() -> str:
    """Create advanced test files with various dates and sizes"""
    test_dir = Path("smart_test_directory")
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
        print("9. Exit")
        
        try:
            choice = input("\nSelect option (1-9): ").strip()
            
            if choice == "1":
                directory = input("Enter directory path: ").strip().strip('"\'')
                if directory:
                    extension_to_category = build_extension_to_category(DEFAULT_CATEGORY_EXTENSIONS.keys())
                    moved = organize_hybrid(Path(directory), extension_to_category, logger)
                    print(f"✓ Smart organization complete! Moved {moved} files")
            
            elif choice == "2":
                directory = input("Enter directory path: ").strip().strip('"\'')
                if directory:
                    date_format = input("Date format (year_month/year_only/full_date): ").strip() or "year_month"
                    moved = organize_by_date(Path(directory), date_format, logger)
                    print(f"✓ Date organization complete! Moved {moved} files")
            
            elif choice == "3":
                directory = input("Enter directory path: ").strip().strip('"\'')
                if directory:
                    moved = organize_by_size(Path(directory), logger)
                    print(f"✓ Size organization complete! Moved {moved} files")
            
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
                test_dir = create_test_files_advanced()
                print(f"✓ Test files created in: {test_dir}")
            
            elif choice == "8":
                if scheduler.scheduler_running:
                    scheduler.stop_scheduler()
                    print("🕒 Scheduler stopped")
                else:
                    scheduler.start_scheduler()
                    print("🕒 Scheduler started")
            
            elif choice == "9":
                scheduler.stop_scheduler()
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please select 1-9.")
                
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
        # Interactive mode
        if args.interactive:
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


if __name__ == "__main__":
    main()


