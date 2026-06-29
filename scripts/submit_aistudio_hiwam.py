#!/usr/bin/env python3
"""Compatibility entrypoint for the isolated AIStudio multi-node submitter."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "aistudio_multinode" / "submit_aistudio_hiwam.py"
    runpy.run_path(str(target), run_name="__main__")
