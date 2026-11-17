"""Run the whole project from start to finish with ONE command.

    python src/run_pipeline.py --name "Your Name"

Steps: generate data -> clean -> EDA -> train/evaluate -> test new tickets
       -> screenshots -> report.pdf

Use --skip-generate to keep your own data/tickets.csv instead of re-creating it.
"""
import argparse
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent


def run(script, *extra):
    print(f"\n{'#' * 70}\n# {script} {' '.join(extra)}\n{'#' * 70}", flush=True)
    result = subprocess.run([sys.executable, script, *extra], cwd=SRC)
    if result.returncode != 0:
        sys.exit(f"\nStep failed: {script}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="Ankita Das", help="your name, printed on report.pdf")
    ap.add_argument("--skip-generate", action="store_true", help="keep the existing data/tickets.csv")
    args = ap.parse_args()

    if not args.skip_generate:
        run("generate_dataset.py")
    run("preprocessing.py")
    run("eda.py")
    run("train.py")
    run("test_new_tickets.py")
    run("make_screenshots.py")
    run("make_report.py", "--name", args.name)
    print("\nAll done. Start the web app with:  python src/app.py")


if __name__ == "__main__":
    main()
