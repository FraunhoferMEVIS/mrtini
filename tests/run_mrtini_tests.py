
"""Run both MRTini and release tests with combined coverage."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fail-under", type=float, default=80,
        help="Minimum combined Python statement/branch coverage (e.g. 90)"
    )
    parser.add_argument(
        "--release", action="store_true",
        help="Compatibility option; regression checks now fail normally"
    )
    args, extra = parser.parse_known_args()

    root = Path(__file__).resolve().parents[1]
    os.chdir(root)

    report_dir = root / "tests/reports/mrtini"
    report_dir.mkdir(parents=True, exist_ok=True)

    required = [
        "src/mrtini/RF_banks.py",
        "src/mrtini/bloch.py",
        "src/sequence/sequence.py",
    ]
    missing = [p for p in required if not (root / p).is_file()]
    if missing:
        parser.error(
            "Extract at MRTini repository root; missing: " + ", ".join(missing)
        )

    common = [
        "--cov",
        "--cov-config=tests/.coveragerc.mrtini",
        "--cov-branch",
    ]

    # 1. MRTini tests: start a fresh coverage measurement
    mrtini_cmd = [
        sys.executable, "-m", "pytest",
        "-c", "tests/pytest.mrtini.ini",
        "tests/mrtini_pytest",
        *common,
        "--cov-report=",
        "--cov-fail-under=0",
        "--junitxml=tests/reports/mrtini/junit-mrtini.xml",
    ]

    # 2. Release tests: append to coverage from MRTini tests
    release_cmd = [
        sys.executable, "-m", "pytest",
        "-c", "tests/pytest.release.ini",
        "tests/release",
        "-m", "not cuda",  # Exclude tests marked @pytest.mark.cuda
        *common,
        "--cov-append",
        "--cov-report=term-missing",
        "--cov-report=html",
        "--cov-report=xml",
        "--cov-report=json",
        "--junitxml=tests/reports/mrtini/junit-release.xml",
        f"--cov-fail-under={args.fail_under}",
    ]

    if args.release:
        mrtini_cmd.append("--runxfail")
        release_cmd.append("--runxfail")

    print("\n" + "=" * 70, flush=True)
    print("RUNNING MRTINI TEST SUITE", flush=True)
    print("=" * 70, flush=True)
    result_mrtini = subprocess.call(mrtini_cmd + extra)

    print("\n" + "=" * 70, flush=True)
    print("RUNNING RELEASE TEST SUITE", flush=True)
    print("=" * 70, flush=True)
    result_release = subprocess.call(release_cmd + extra)

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"MRTini tests : {'PASS' if result_mrtini == 0 else 'FAIL'}")
    print(f"Release tests: {'PASS' if result_release == 0 else 'FAIL'}")
    print("Coverage     : Combined report from both suites")
    print("=" * 70)

    return int(result_mrtini != 0 or result_release != 0)


if __name__ == "__main__":
    raise SystemExit(main())
