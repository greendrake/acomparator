# acomparator

CLI tool for comparing two audio files to determine if they are from the same source.

Typical use cases:

- You have several different audio files of ostensibly the same audio track. They may differ by quality, start offset, level, some mastering differences — all this does not make the tracks themselves different as far as you are concerned. However, there could be actual mixing differences e.g. different effects, samples, or outright some chunks edited out/added. You need to find out which of the two cases it is, and if there are mixing differences, what and where they exactly are.
- You have two audio tracks for a movie, and suspect that they may actually be slightly different e.g. some words in one of them could be silenced/replaced. You need to find out whether this is the case, and what/where those different fragments, if any, are.

## Features

- Detects if two audio files are the **same source** (different mastering/quality)
- Detects if files are **completely different**
- Detects **partial matches** (e.g., sections cut or replaced)
- Extracts differing sections to separate WAV files
- Supports any audio format FFmpeg can decode (WAV, MP3, FLAC, OGG, etc.)

## System Requirements

- Linux (x86_64 or aarch64)
- FFmpeg
- libchromaprint (for audio fingerprinting)

### Install dependencies (Debian/Ubuntu)

```bash
sudo apt install ffmpeg libchromaprint-tools
```

### Install dependencies (Fedora/RHEL)

```bash
sudo dnf install ffmpeg chromaprint-tools
```

### Install dependencies (Arch)

```bash
sudo pacman -S ffmpeg chromaprint
```

## Installation

### Option 1: Pre-built executable

Download the executable for your architecture from releases and make it executable:

```bash
chmod +x acomparator
./acomparator --help
```

### Option 2: Build from source

Requirements: Python 3.11+

```bash
# Clone the repository
git clone <repo-url>
cd acomparator

# Build executable
chmod +x build.sh
./build.sh

# Executable is at dist/acomparator
```

### Option 3: Install as Python package

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install
pip install -e .

# Run
acomparator --help
```

## Testing

```bash
# Run tests against Python source
./test.sh

# Build executable and run tests against it
./test.sh --executable
```

## Development

### Code Formatting

This project uses [ruff](https://docs.astral.sh/ruff/) for code formatting and linting.

```bash
# Check for issues (used in CI)
./lint.sh

# Automatically fix issues and format code
./lint.sh --fix
```

### Project Scripts

| Script     | Description                                               |
| ---------- | --------------------------------------------------------- |
| `build.sh` | Build standalone executable using PyInstaller             |
| `test.sh`  | Run test suite (use `--executable` to test built binary)  |
| `lint.sh`  | Run code formatting and linting (use `--fix` to auto-fix) |

## Usage

```
acomparator [OPTIONS] FILE_A FILE_B
```

### Arguments

| Argument | Description                  |
| -------- | ---------------------------- |
| `FILE_A` | First audio file to compare  |
| `FILE_B` | Second audio file to compare |

### Options

| Option                        | Default    | Description                                                      |
| ----------------------------- | ---------- | ---------------------------------------------------------------- |
| `--output-dir PATH`           | `./output` | Directory to save extracted difference segments                  |
| `--same-threshold FLOAT`      | `0.85`     | Minimum similarity for "same source" classification              |
| `--different-threshold FLOAT` | `0.30`     | Maximum similarity for "completely different" classification     |
| `--min-diff-duration FLOAT`   | `0.1`      | Minimum duration (seconds) for a difference to be reported       |
| `--diff-threshold FLOAT`      | `0.8`      | Similarity threshold below which frames are considered different |
| `--no-fingerprint`            | off        | Skip chromaprint fingerprint pre-check                           |
| `--help`                      |            | Show help message                                                |

## Examples

### Basic comparison

```bash
acomparator original.wav remastered.wav
```

### Compare with custom output directory

```bash
acomparator file1.mp3 file2.flac --output-dir ./differences
```

### Adjust similarity thresholds

```bash
acomparator file1.wav file2.wav --same-threshold 0.90 --different-threshold 0.25
```

### Skip fingerprint check (faster for known-similar files)

```bash
acomparator file1.wav file2.wav --no-fingerprint
```

### Less sensitive detection (for files with different mastering)

If comparing files with different mastering that triggers false positives, use a lower threshold:

```bash
acomparator file1.wav file2.wav --diff-threshold 0.5
```

Note: Lower `--diff-threshold` values are less sensitive and avoid false positives from mastering differences, but may miss subtle edits.

## Output

The tool outputs JSON to stdout with the comparison result.

### Same source

Files are from the same source with only quality/mastering differences:

```json
{
  "result": "same_source",
  "overall_similarity": 0.93,
  "alignment_offset_seconds": -0.14,
  "fingerprint_match": true,
  "differences": []
}
```

### Completely different

Files have no resemblance:

```json
{
  "result": "completely_different",
  "overall_similarity": 0.0,
  "alignment_offset_seconds": 22.52,
  "fingerprint_match": false,
  "differences": []
}
```

### Partial match

Files are mostly the same but have sections that differ (e.g., a section was cut):

```json
{
  "result": "partial_match",
  "overall_similarity": 0.78,
  "alignment_offset_seconds": 0.0,
  "fingerprint_match": true,
  "differences": [
    {
      "start_seconds": 98.43,
      "end_seconds": 138.4,
      "duration_seconds": 39.97,
      "file_a_segment": "output/diff_001_a.wav",
      "file_b_segment": "output/diff_001_b.wav"
    }
  ]
}
```

For partial matches, the differing sections are extracted to WAV files in the output directory.

## Output Fields

| Field                      | Description                                                               |
| -------------------------- | ------------------------------------------------------------------------- |
| `result`                   | Classification: `same_source`, `completely_different`, or `partial_match` |
| `overall_similarity`       | Average frame similarity score (0.0 to 1.0)                               |
| `alignment_offset_seconds` | Time offset used to align the files (positive = file B starts later)      |
| `fingerprint_match`        | Whether audio fingerprints indicate same source                           |
| `differences`              | Array of differing sections (only for `partial_match`)                    |

## Exit Codes

| Code | Meaning                                      |
| ---- | -------------------------------------------- |
| 0    | Success (any result)                         |
| 1    | Error (file not found, invalid format, etc.) |

## How It Works

1. **Fingerprint check**: Quick comparison using Chromaprint audio fingerprints
2. **Alignment**: Cross-correlation on chroma features to find time offset
3. **Similarity**: MFCC extraction with per-frame cosine similarity
4. **Classification**: Based on overall similarity and detected differences
5. **Extraction**: For partial matches, differing sections are saved as WAV files

## Memory Efficiency

The tool uses chunked processing for memory-efficient comparison of large files:

- Audio is processed in 60-second chunks with 5-second overlap
- Only first 120 seconds are loaded for alignment detection
- Peak memory usage is ~200MB regardless of file duration
- Can handle multi-hour files (podcasts, DJ mixes, audiobooks)

## Limitations

- Requires FFmpeg for non-WAV formats
- Executable is architecture-specific (build separately for x86_64 vs aarch64)
- Very short files (< 10 seconds) may have reduced accuracy
- Extreme tempo changes between files are not handled

## License

MIT
