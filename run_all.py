"""
run_all.py — Run the whole analysis pipeline in order, data quality first
=========================================================================

KIE4033 Assignment 1 · SDG 14 overfishing analysis (Global -> Southeast Asia)

    python run_all.py                  # every step
    python run_all.py --from 3         # resume at step 3 (step 0 must have passed before)

Order (the data-analytics lifecycle: understand and check the data before analysing it)
    0  0-data-quality/clean_data.py         check every raw file, clean it, build the tidy
                                            tables; STOPS the pipeline if any check fails
    0  0-data-quality/quality_tests.py      validation against alternative sources + integrity
    1  1-visualisation/visualise.py         exploratory figures (from the checked data)
    2  2-statistical-tests/stat_tests.py    course-method tests
    3  3-hypothesis-tests/hypothesis_tests.py   hypothesis tests, 7 iterations
    4  4-overview/overview.py               report-level summaries + figure catalogue
    4  dashboard/build_dashboard.py         offline interactive results page

Each step is run as its own process; the first one that fails stops the run with its
exit status. Steps 1–4 also refuse to start on their own unless step 0 has passed on
the current data/raw/ files (src/loaders.py: require_checked_data).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [(0, "0-data-quality/clean_data.py"), (0, "0-data-quality/quality_tests.py"),
         (1, "1-visualisation/visualise.py"), (2, "2-statistical-tests/stat_tests.py"),
         (3, "3-hypothesis-tests/hypothesis_tests.py"), (4, "4-overview/overview.py"),
         (4, "dashboard/build_dashboard.py")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the pipeline in order, data quality first.")
    parser.add_argument("--from", dest="start", type=int, default=0, choices=range(5),
                        help="first step to run (0–4)")
    args = parser.parse_args()
    for step, script in STEPS:
        if step < args.start:
            continue
        print(f"\n=== step {step}: {script}", flush=True)
        t0 = time.time()
        code = subprocess.call([sys.executable, str(ROOT / script)], cwd=ROOT)
        if code != 0:
            print(f"\nStopped: {script} exited with status {code}. Later steps were not run.")
            sys.exit(code)
        print(f"--- done in {time.time() - t0:.0f} s", flush=True)
    print("\nAll steps finished.")


if __name__ == "__main__":
    main()
