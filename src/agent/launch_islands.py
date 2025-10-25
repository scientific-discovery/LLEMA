import os
import sys
import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Launch multi-island run.")
    parser.add_argument("--islands", type=int, default=4, help="Number of islands to launch.")
    parser.add_argument("--candidates", type=int, default=2, help="Candidates per generation call.")
    parser.add_argument("--islands-dir", type=str, default="", help="Shared islands dir for memories.")
    parser.add_argument("--workdir", type=str, default="", help="Working directory to run from.")
    args = parser.parse_args()

    if args.workdir:
        os.chdir(args.workdir)

    if not args.islands_dir:
        base_dir = Path(__file__).resolve().parent
        islands_dir = base_dir / "runs" / "islands"
    else:
        islands_dir = Path(args.islands_dir)

    islands_dir.mkdir(parents=True, exist_ok=True)

    procs = []
    for i in range(args.islands):
        env = os.environ.copy()
        env["ISLAND_ID"] = str(i)
        env["ISLANDS_DIR"] = str(islands_dir)
        cmd = [sys.executable, "-m", "agent.main", str(args.candidates)]
        procs.append(subprocess.Popen(cmd, env=env))

    code = 0
    try:
        for p in procs:
            p.wait()
            if p.returncode != 0:
                code = p.returncode
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
    sys.exit(code)


if __name__ == "__main__":
    main()


