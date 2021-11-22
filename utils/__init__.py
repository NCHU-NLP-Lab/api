import json
from pathlib import Path

from .export import delete_later, export_file

__all__ = [
    "delete_later",
    "export_file",
]


def load_examples():
    """Load examples if examples exist"""
    example_file = Path(__file__).parent.parent / "data" / "examples.json"
    if example_file.exists():
        with open(example_file, "r") as f:
            return json.load(f)
