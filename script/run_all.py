import argparse
import subprocess
from pathlib import Path


def main(args):
    script_path = Path(__file__).parent / "all.sh"
    if "," in args.data:
        for data in args.data.split(","):
            cmd = [
                str(script_path),
                data,
            ]
            subprocess.run(cmd)
    else:
        cmd = [
            str(script_path),
            args.data,
        ]
        subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)

    args = parser.parse_args()
    main(args)
