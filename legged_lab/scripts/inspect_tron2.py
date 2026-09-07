"""Usage: python -m legged_lab.scripts.inspect_tron2 --description-root PATH."""

import argparse
import json
from pathlib import Path

from legged_lab.utils.tron2_description import inspect_variant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--description-root", required=True)
    parser.add_argument("--variant", help="Omit to inspect all variants")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    variants = [args.variant] if args.variant else sorted(
        p.name for p in (Path(args.description_root) / "tron2").iterdir() if p.is_dir()
    )
    reports = [inspect_variant(args.description_root, variant) for variant in variants]
    serialized = json.dumps(reports, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    for report in reports:
        print(f"{report['variant']}: {report['movable_joint_count']} joints; "
              f"USD={report['usd_exists']}; MJCF={report['mjcf_exists']}; "
              f"missing meshes={len(report['missing_meshes'])}")
    if any(report["missing_meshes"] for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
