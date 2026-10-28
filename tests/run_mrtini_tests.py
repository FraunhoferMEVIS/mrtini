"""Run from any working directory using the repository's Python environment."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fail-under", type=float, default=90,
                        help="Minimum combined Python statement/branch coverage (e.g. 90)")
    parser.add_argument("--release", action="store_true",
                        help="Compatibility option; regression checks now fail normally")
    args, extra = parser.parse_known_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    (root / "tests/reports/mrtini").mkdir(parents=True, exist_ok=True)
    required = ["src/mrtini/RF_banks.py", "src/mrtini/bloch.py", "src/sequence/sequence.py"]
    missing = [p for p in required if not (root / p).is_file()]
    if missing:
        parser.error("Extract at MRtini repository root; missing: " + ", ".join(missing))
    cmd = [sys.executable, "-m", "pytest", "-c", "tests/pytest.mrtini.ini", "tests/mrtini_pytest",
           "--cov", "--cov-config=tests/.coveragerc.mrtini", "--cov-branch",
           "--cov-report=term-missing", "--cov-report=html", "--cov-report=xml",
           "--cov-report=json", "--junitxml=tests/reports/mrtini/junit.xml",
           f"--cov-fail-under={args.fail_under}"]
    if args.release: cmd.append("--runxfail")
    return subprocess.call(cmd + extra)

if __name__ == "__main__":
    raise SystemExit(main())
