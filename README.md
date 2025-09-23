# Smart File Organizer - Phase 3

Advanced file organization with smart features and intelligent categorization.

## 🚀 Features

### Core Organization
- **Smart Categorization**: Classifies into Documents, Images, Videos, Audio, Archives, Code, and Others
- **Multiple Organization Modes**:
  - Extension-based (default)
  - Date-based (by creation date)
  - Size-based (Small/Medium/Large files)
  - Hybrid (Type + Date combination)
- **Dynamic Folder Creation**: Creates destination folders automatically
- **Safe File Moving**: Handles name collisions with intelligent renaming
- **Dry-run Mode**: Preview changes before execution

### Smart Features
- **🔍 Duplicate Detection**: Find duplicate files using MD5 hashing
- **🗑️ Interactive Duplicate Removal**: Choose which duplicates to keep
- **📊 Comprehensive Statistics**: Detailed directory analysis and reporting
- **🕒 Automatic Scheduler**: Schedule daily/weekly organization
- **🎮 Interactive Mode**: User-friendly menu interface
- **🧪 Test File Generator**: Create test files for demonstration

### Advanced Capabilities
- **Recursive Scanning**: Process subdirectories (skips category folders)
- **Flexible Date Formats**: Year/Month, Year only, or Full date organization
- **Size Categories**: Small (<1MB), Medium (1-100MB), Large (>100MB)
- **Hybrid Organization**: Group by type, then by date (e.g., Images/2024/January/)
- **Comprehensive Logging**: Console and file logging with configurable levels

## 📋 Requirements

- Python 3.8+
- Optional: `schedule` library for automatic scheduling (`pip install schedule`)

## 🛠️ Installation

1. Clone or download the repository
2. Install optional dependencies:
   ```bash
   pip install schedule
   ```

## 🚀 Usage

### Graphical User Interface (GUI)

```bash
# Launch the basic Tkinter GUI
python organize_files.py --gui
```

Optional drag & drop support:

```bash
pip install tkinterdnd2
```

If drag-and-drop is available, you can drop a folder onto the folder field.

### Command Line Interface

#### Basic Organization
```bash
# Organize current directory (dry-run first)
python organize_files.py --path . --dry-run

# Organize specific directory
python organize_files.py --path C:\Users\Prakash\Downloads

# Organize with specific categories
python organize_files.py --path . --categories Documents,Images,Videos
```

#### Smart Organization Modes
```bash
# Date-based organization
python organize_files.py --path . --mode date --date-format year_month

# Size-based organization
python organize_files.py --path . --mode size

# Hybrid organization (type + date)
python organize_files.py --path . --mode hybrid
```

#### Duplicate Management
```bash
# Find duplicates
python organize_files.py --path . --find-duplicates

# Interactive duplicate removal
python organize_files.py --path . --remove-duplicates
```

#### Statistics and Analysis
```bash
# Show directory statistics
python organize_files.py --path . --stats
```

#### Automatic Scheduling
```bash
# Schedule daily organization at 2:00 AM
python organize_files.py --path . --schedule daily --time 02:00 --mode hybrid

# Schedule weekly organization
python organize_files.py --path . --schedule weekly --time 09:00 --mode date
```

#### Interactive Mode
```bash
# Launch interactive menu
python organize_files.py --interactive

# Or use the CLI alias flag
python organize_files.py --cli
```

### Interactive Mode Menu

When using `--interactive`, you'll get a user-friendly menu with options:

1. **Smart Organize** (Extension + Date)
2. **Organize by Date Only**
3. **Organize by Size Only**
4. **Find & Remove Duplicates**
5. **Schedule Auto-Organization**
6. **Directory Statistics**
7. **Create Test Files**
8. **Start/Stop Scheduler**
9. **Exit**

## 📁 File Categories and Extensions

### Documents
`.pdf`, `.doc`, `.docx`, `.txt`, `.rtf`, `.odt`, `.xls`, `.xlsx`, `.ppt`, `.pptx`, `.csv`, `.md`, `.json`, `.xml`, `.yaml`, `.yml`

### Images
`.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.tiff`, `.tif`, `.webp`, `.svg`, `.heic`

### Videos
`.mp4`, `.mkv`, `.avi`, `.mov`, `.wmv`, `.flv`, `.webm`, `.m4v`

### Audio
`.mp3`, `.wav`, `.flac`, `.aac`, `.ogg`, `.m4a`, `.wma`, `.aiff`

### Archives
`.zip`, `.rar`, `.7z`, `.tar`, `.gz`, `.bz2`, `.xz`

### Code
`.py`, `.js`, `.ts`, `.tsx`, `.jsx`, `.java`, `.c`, `.cpp`, `.cs`, `.go`, `.rs`, `.rb`, `.php`, `.swift`, `.kt`, `.m`, `.html`, `.css`, `.scss`, `.sass`, `.sh`, `.ps1`, `.bat`

### Others
Any unrecognized file extension

## ⚙️ Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--path` | Directory to organize | Current directory (`.`) |
| `--dry-run` | Preview changes without moving files | False |
| `--recursive` | Scan subdirectories recursively | False |
| `--categories` | Comma-separated categories to enable | All categories |
| `--mode` | Organization mode: `extension`, `date`, `size`, `hybrid` | `extension` |
| `--date-format` | Date format: `year_month`, `year_only`, `full_date` | `year_month` |
| `--find-duplicates` | Find and report duplicate files | False |
| `--remove-duplicates` | Interactive duplicate removal | False |
| `--stats` | Show comprehensive directory statistics | False |
| `--schedule` | Schedule organization: `daily`, `weekly` | None |
| `--time` | Time for scheduled runs (HH:MM format) | `02:00` |
| `--interactive` | Launch interactive mode | False |
| `--gui` | Launch basic Tkinter GUI | False |
| `--cli` | Alias of --interactive for CLI menu | False |
| `--log-file` | Path to log file | `organize_files.log` |
| `--log-level` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` | `INFO` |

## 📊 Organization Examples

### Extension-based (Default)
```
Downloads/
├── Documents/
│   ├── report.pdf
│   └── spreadsheet.xlsx
├── Images/
│   ├── photo1.jpg
│   └── photo2.png
└── Videos/
    └── movie.mp4
```

### Date-based
```
Downloads/
├── 2024/
│   ├── January/
│   │   ├── old_file.pdf
│   │   └── document.docx
│   └── February/
│       └── recent_file.txt
```

### Size-based
```
Downloads/
├── Small_Files/
│   ├── config.txt
│   └── readme.md
├── Medium_Files/
│   └── presentation.pptx
└── Large_Files/
    └── video.mp4
```

### Hybrid (Type + Date)
```
Downloads/
├── Documents/
│   └── 2024/
│       └── January/
│           └── report.pdf
├── Images/
│   └── 2024/
│       └── February/
│           └── photo.jpg
```

## 🔧 Advanced Features

### Duplicate Detection
- Uses MD5 hashing for accurate duplicate detection
- Chunked reading for large files
- Interactive removal with user choice
- Space savings reporting

### Statistics Reporting
- Total file count and size
- File type distribution
- Size category breakdown
- Oldest and newest file information

### Automatic Scheduling
- Background thread execution
- Daily or weekly scheduling
- Configurable time settings
- Graceful shutdown with Ctrl+C

## ⚠️ Important Notes

- **Always use `--dry-run` first** to preview changes
- Files already in category folders are skipped
- Name collisions are resolved with numeric suffixes
- Scheduler requires `schedule` library installation
- Interactive mode provides the easiest way to use all features

## 🐛 Troubleshooting

### Common Issues
1. **Permission Errors**: Run as administrator if needed
2. **Schedule Library Missing**: Install with `pip install schedule`
3. **Large File Processing**: Duplicate detection may take time for large files
4. **Path Issues**: Use forward slashes or escaped backslashes in paths

### Logging
- Check `organize_files.log` for detailed execution logs
- Use `--log-level DEBUG` for verbose output
- Logs include timestamps and operation details

## 📈 Performance

- **Small directories** (< 1000 files): Near-instant processing
- **Medium directories** (1000-10000 files): 1-5 seconds
- **Large directories** (> 10000 files): 10-60 seconds depending on file sizes
- **Duplicate detection**: Additional time based on file sizes and count

## 🤝 Contributing

This is a personal project, but suggestions and improvements are welcome!

## 📄 License

This project is for personal use. Feel free to modify and distribute as needed.