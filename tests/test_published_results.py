import subprocess
import sys
from pathlib import Path


def test_archived_tables_match_reported_values() -> None:
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "scripts/validate_published_results.py"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "internally consistent" in completed.stdout
