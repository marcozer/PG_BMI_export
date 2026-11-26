#!/usr/bin/env python3
"""Run all HBSN revision analyses and exports."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from scripts.generate_revision import main as generate_revision


if __name__ == "__main__":
    generate_revision()
