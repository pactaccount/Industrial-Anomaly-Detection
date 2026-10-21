"""Shared test fixtures.

Every test run gets its own data directory, so tests never touch a real
database or leave artefacts behind.
"""

from __future__ import annotations

import os
import tempfile

import pytest


@pytest.fixture(scope="session", autouse=True)
def isolated_data_dir():
    """Point the whole session at a throwaway data directory."""
    path = tempfile.mkdtemp(prefix="argus_pytest_")
    os.environ["ARGUS_DATA_DIR"] = path
    os.environ.setdefault("ARGUS_DATABASE_URL", f"sqlite:///{path}/test.db")
    yield path
