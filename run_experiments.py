from __future__ import annotations

import argparse
from datetime import datetime

from ppm_ad.runner import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PPM-AD experiments on MVTec AD")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", choices=("fewshot", "ablation", "drift", "k_sweep", "all"), default="fewshot")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    run_name = args.run_name or f"{args.mode}-{datetime.now():%Y%m%d-%H%M%S}"
    output = run(args.config, args.mode, run_name)
    print(f"Results written to: {output}")


if __name__ == "__main__":
    main()


