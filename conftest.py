"""Ensures the project root is on sys.path so tests can import our packages.
This is a pytest convention — conftest.py is auto-loaded before tests run.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
