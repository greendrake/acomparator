"""JSON output serialization for comparison reports."""

import json
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any

from acomparator.models import ComparisonReport


class ReportEncoder(json.JSONEncoder):
    """Custom JSON encoder for ComparisonReport."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def report_to_json(report: ComparisonReport) -> str:
    """
    Serialize a ComparisonReport to JSON string.

    Args:
        report: The comparison report to serialize.

    Returns:
        JSON string representation.
    """
    data = asdict(report)
    return json.dumps(data, cls=ReportEncoder, indent=2)


def error_to_json(message: str) -> str:
    """
    Serialize an error message to JSON string.

    Args:
        message: The error message.

    Returns:
        JSON string with error field.
    """
    return json.dumps({"error": message}, indent=2)
