# Smart File Organizer

Advanced file organization with smart features, intelligent categorization, and a complete enhanced graphical user interface.

## Features

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

### Enhanced GUI Features (v2.1)
- 🎨 Complete Layout: Both left and right panels with modern design
- **🔧 Fixed TTK Issues**: Resolved font parameter and geometry manager conflicts
- **👁️ All Buttons Visible**: Every action button is properly displayed
- **📜 Scrollable Interface**: Handle large file lists with smooth scrolling
- **🎯 Drag & Drop Support**: Drop folders directly onto the interface
- **🔍 Preview Mode**: See planned changes before applying them
- **↩️ Undo/Redo Operations**: Revert or reapply file organization actions
- **📊 Real-time Progress**: Detailed progress tracking with file counts and ETA
- **📈 Live Statistics**: Real-time file processing statistics
- **🎛️ Multiple Organization Modes**: Easy switching between extension, date, size, and hybrid modes
- **🔍 Enhanced Duplicate Detection**: Visual duplicate file identification
- **📋 Operation Summary**: Detailed summary of all planned operations

### Advanced Capabilities
- **Recursive Scanning**: Process subdirectories (skips category folders)
- **Flexible Date Formats**: Year/Month, Year only, or Full date organization
- **Size Categories**: Small (<1MB), Medium (1-100MB), Large (>100MB)
- **Hybrid Organization**: Group by type, then by date (e.g., Images/2024/January/)
- **Comprehensive Logging**: Console and file logging with configurable levels
- **Undo/Redo System**: Persistent operation history with JSON storage
- **Progress Tracking**: Real-time progress with detailed statistics and ETA

## 📋 Requirements

- Python 3.8+
- Optional: `schedule` library for automatic scheduling (`pip install schedule`)
- Optional: `tkinterdnd2` library for enhanced drag & drop support (`pip install tkinterdnd2`)

## 🛠️ Installation

1. Clone or download the repository
2. Install optional dependencies:
   ```bash
   # For automatic scheduling
   pip install schedule
   
   # For enhanced drag & drop support
   pip install tkinterdnd2
   ```

## 🚀 Usage

### Enhanced Graphical User Interface (GUI)

```bash
# Launch the Enhanced GUI 
python organize_files.py --gui
```

The Enhanced GUI includes:

#### 🎨 **Complete Layout**
- **Left Panel**: Folder selection, organization options, actions, and progress tracking
- **Right Panel**: Preview changes, operation summary, and detailed file information

#### 🎯 **Key Features**
- **Drag & Drop**: Drop folders directly onto the folder field (requires `tkinterdnd2`)
- **Preview Mode**: See all planned file moves before applying them
- **Undo/Redo**: Revert or reapply organization operations
- **Real-time Progress**: Live progress tracking with file counts, sizes, and ETA
- **Multiple Modes**: Easy switching between extension, date, size, and hybrid organization
- **Enhanced Duplicate Detection**: Visual identification of duplicate files
- **Operation Summary**: Detailed summary of all planned operations

#### 🔧 **Installation for Full Features**
```bash
# For enhanced drag & drop support
pip install tkinterdnd2
```

#### 🎮 **GUI Usage**
1. **Select Folder**: Use the browse button or drag & drop a folder
2. **Choose Mode**: Select organization mode (Extension, Date, Size, or Hybrid)
3. **Configure Options**: Set date format if using date-based organization
4. **Preview Changes**: Enable preview mode to see planned operations
5. **Start Organization**: Click "Start Organization" to begin
6. **Review Preview**: Check the right panel for planned file moves
7. **Apply Changes**: Click "Apply Changes" to execute the organization
8. **Undo if Needed**: Use the Undo button to revert changes

#### 🔄 **Fallback Support**
If the Enhanced GUI can't start (e.g., system Tk components are missing), the app falls back to the interactive command‑line mode so you can continue working.

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

When you run with `--interactive`, you get a straightforward menu focused on the tasks you'll actually use:

1. Smart Organize (Extension + Date)
2. Organize by Date Only
3. Organize by Size Only
4. Find & Remove Duplicates
5. Schedule Auto-Organization
6. Directory Statistics
7. Create Test Files
8. Start/Stop Scheduler
9. Undo last operation
10. Redo last operation
11. Open Enhanced GUI
12. Exit

Notes:
- Undo/Redo applies to organize actions performed in this session. If you undo, you can immediately redo the same operation.
- The test file generator asks for a folder; press Enter to use the current directory.

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
| `--date-format` | Date format: `year`, `year_month`, `full` | `year_month` |
| `--find-duplicates` | Find and report duplicate files | False |
| `--remove-duplicates` | Interactive duplicate removal | False |
| `--stats` | Show comprehensive directory statistics | False |
| `--schedule` | Schedule organization: `daily`, `weekly` | None |
| `--time` | Time for scheduled runs (HH:MM format) | `02:00` |
| `--interactive` | Launch interactive mode | False |
| `--gui` | Launch Enhanced GUI (v2.1) with fallback to basic GUI | False |
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

### Enhanced GUI Features (v2.1)
- **Complete Layout**: Two-panel design with left controls and right preview
- **Modern Styling**: Clean, professional interface with proper fonts and colors
- **Drag & Drop**: Direct folder dropping for quick selection
- **Preview Mode**: See all planned operations before execution
- **Undo/Redo System**: Persistent operation history with JSON storage
- **Real-time Progress**: Live updates with file counts, sizes, and ETA
- **Enhanced Statistics**: Visual display of file processing statistics
- **Operation Summary**: Detailed breakdown of all planned moves

### Duplicate Detection
- Uses MD5 hashing for accurate duplicate detection
- Chunked reading for large files
- Interactive removal with user choice
- Space savings reporting
- Enhanced GUI visualization of duplicate groups

### Statistics Reporting
- Total file count and size
- File type distribution
- Size category breakdown
- Oldest and newest file information
- Real-time processing statistics in GUI

### Automatic Scheduling
- Background thread execution
- Daily or weekly scheduling
- Configurable time settings
- Graceful shutdown with Ctrl+C

## ⚠️ Important Notes

- **Enhanced GUI includes built-in preview mode** - no need for `--dry-run` when using GUI
- **Always use preview mode in GUI** to see planned changes before applying
- Files already in category folders are skipped
- Name collisions are resolved with numeric suffixes
- Scheduler requires `schedule` library installation
- Enhanced GUI provides the easiest way to use all features
- Undo/Redo operations are automatically saved and persist between sessions
- Drag & drop requires `tkinterdnd2` library installation

## 🐛 Troubleshooting

### Common Issues
1. **Permission Errors**: Run as administrator if needed
2. **Schedule Library Missing**: Install with `pip install schedule`
3. **Drag & Drop Not Working**: Install with `pip install tkinterdnd2`
4. **Enhanced GUI Not Starting**: Check console for error messages, falls back to basic GUI
5. **Large File Processing**: Duplicate detection may take time for large files
6. **Path Issues**: Use forward slashes or escaped backslashes in paths
7. **Undo/Redo Issues**: Check if `file_organizer_undo.json` is writable

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