## File Organizer

Organize files in a directory by category based on file extensions.

### Features
- Classifies into Documents, Images, Videos, Audio, Archives, Code, and Others
- Creates destination folders dynamically
- Safe moves with collision handling (adds "(1)", "(2)", ...)
- Dry-run mode to preview changes
- Optional recursive scan (skips category folders and common system dirs)
- Built-in logging to console and file

### Requirements
- Python 3.8+

### Usage
From the project folder in PowerShell:

```bash
python organize_files.py --path . --dry-run
python organize_files.py --path C:\Users\anjan\Downloads
python organize_files.py --path . --recursive
python organize_files.py --path . --categories Documents,Images,Videos
python organize_files.py --path . --dry-run --log-level DEBUG
python organize_files.py --path . --log-file C:\\temp\\organize.log
```

Options:
- `--path`: Directory to organize (default: current directory)
- `--dry-run`: Preview planned moves without changing files
- `--recursive`: Scan subdirectories (skips category folders)
- `--categories`: Comma-separated list of categories to enable. Unknown extensions go to Others.
- `--log-file`: Path to write logs (default: `organize_files.log`)
- `--log-level`: Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` (default: `INFO`)

### Categories and Extensions
Default mapping (subset):

- Documents: `.pdf, .doc, .docx, .txt, .rtf, .odt, .xls, .xlsx, .ppt, .pptx, .csv, .md, .json, .xml, .yaml, .yml`
- Images: `.jpg, .jpeg, .png, .gif, .bmp, .tiff, .tif, .webp, .svg, .heic`
- Videos: `.mp4, .mkv, .avi, .mov, .wmv, .flv, .webm, .m4v`
- Audio: `.mp3, .wav, .flac, .aac, .ogg, .m4a, .wma, .aiff`
- Archives: `.zip, .rar, .7z, .tar, .gz, .bz2, .xz`
- Code: `.py, .js, .ts, .tsx, .jsx, .java, .c, .cpp, .cs, .go, .rs, .rb, .php, .swift, .kt, .m, .html, .css, .scss, .sass, .sh, .ps1, .bat`
- Others: any unrecognized extension

### Notes
- Files already inside category folders (e.g., `Documents/`) are left in place.
- On name collisions, the script appends a numeric suffix.
- Use `--dry-run` first to verify planned changes.

### Mid Project Review Checklist
- Script correctly identifies and classifies file types: Yes
- Folders are created and files organized successfully: Yes (run without `--dry-run`)


