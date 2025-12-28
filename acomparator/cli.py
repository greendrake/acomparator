"""CLI interface for audio comparison tool."""

import sys
from pathlib import Path

import click

from acomparator.config import Config
from acomparator.core.comparison import compare_audio_files
from acomparator.output import error_to_json, report_to_json


@click.command()
@click.argument("file_a", type=click.Path(exists=True, path_type=Path))
@click.argument("file_b", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("./output"),
    help="Directory to save extracted difference segments.",
)
@click.option(
    "--same-threshold",
    type=float,
    default=0.85,
    help="Minimum similarity for 'same source' classification.",
)
@click.option(
    "--different-threshold",
    type=float,
    default=0.30,
    help="Maximum similarity for 'completely different' classification.",
)
@click.option(
    "--min-diff-duration",
    type=float,
    default=0.1,
    help="Minimum duration (seconds) for a difference to be reported.",
)
@click.option(
    "--diff-threshold",
    type=float,
    default=0.8,
    help="Similarity threshold below which frames are considered different.",
)
@click.option(
    "--no-fingerprint",
    is_flag=True,
    default=False,
    help="Skip chromaprint fingerprint pre-check.",
)
def main(
    file_a: Path,
    file_b: Path,
    output_dir: Path,
    same_threshold: float,
    different_threshold: float,
    min_diff_duration: float,
    diff_threshold: float,
    no_fingerprint: bool,
) -> None:
    """
    Compare two audio files for similarity.

    FILE_A and FILE_B are the audio files to compare.

    Outputs JSON with comparison result: same_source, completely_different,
    or partial_match (with extracted difference segments).
    """
    config = Config(
        same_threshold=same_threshold,
        different_threshold=different_threshold,
        min_diff_duration=min_diff_duration,
        diff_threshold=diff_threshold,
        use_fingerprint=not no_fingerprint,
        output_dir=output_dir,
    )

    try:
        report = compare_audio_files(file_a, file_b, config)
        print(report_to_json(report))
        sys.exit(0)
    except Exception as e:
        print(error_to_json(str(e)))
        sys.exit(1)


if __name__ == "__main__":
    main()
