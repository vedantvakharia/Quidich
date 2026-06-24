#!/usr/bin/env python3
"""Run the Group 1 Phase 0 readiness audit on the cricket drive payload."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pose_estimation.cricket.blockers import extract_external_blockers
from pose_estimation.cricket.calibration import audit_calibration
from pose_estimation.cricket.contract import contract_report
from pose_estimation.cricket.dataset import discover_dataset
from pose_estimation.cricket.events import inspect_events_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", default="drive", help="Path to the moved drive payload")
    parser.add_argument("--run-id", default="phase0-local")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--expected-frame-count", type=int, default=600)
    parser.add_argument("--skip-image-dimensions", action="store_true")
    parser.add_argument(
        "--fail-on-internal-errors",
        action="store_true",
        help="Exit non-zero if dataset/calibration/events/contract checks fail",
    )
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def report_errors(report: dict[str, Any]) -> list[str]:
    return [str(error) for error in report.get("errors", [])]


def build_readiness(
    *,
    run_id: str,
    drive_root: Path,
    reports: dict[str, dict[str, Any]],
    external_blockers: dict[str, Any],
) -> dict[str, Any]:
    internal_errors = []
    for name, report in reports.items():
        for error in report_errors(report):
            internal_errors.append({"report": name, "error": error})

    internal_status = "pass" if not internal_errors else "fail"
    external_status = external_blockers.get("external_readiness", "blocked")
    if internal_status == "pass" and external_status == "ready":
        phase0_status = "complete"
    elif internal_status == "pass":
        phase0_status = "technically_complete_external_blocked"
    else:
        phase0_status = "internal_failed"

    return {
        "schema_version": "phase0_readiness/v1",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "drive_root": str(drive_root),
        "phase0_status": phase0_status,
        "internal_status": internal_status,
        "external_status": external_status,
        "internal_error_count": len(internal_errors),
        "external_blocker_count": len(external_blockers.get("blockers", [])),
        "internal_errors": internal_errors,
        "reports": {
            "dataset_inventory": "dataset_inventory.json",
            "calibration_report": "calibration_report.json",
            "events_pipeline_report": "events_pipeline_report.json",
            "contract_report": "contract_report.json",
            "external_blockers": "external_blockers.json",
        },
    }


def main() -> int:
    args = parse_args()
    drive_root = resolve_path(args.drive_root)
    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir
        else ROOT / "benchmarks" / "runs" / args.run_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_inventory = discover_dataset(
        drive_root,
        expected_frame_count=args.expected_frame_count,
        inspect_dimensions=not args.skip_image_dimensions,
    )
    calibration_report = audit_calibration(drive_root)
    events_pipeline_report = inspect_events_pipeline(drive_root)
    group1_contract_report = contract_report()
    external_blockers = extract_external_blockers(drive_root)

    reports = {
        "dataset_inventory": dataset_inventory,
        "calibration_report": calibration_report,
        "events_pipeline_report": events_pipeline_report,
        "contract_report": group1_contract_report,
    }
    readiness = build_readiness(
        run_id=args.run_id,
        drive_root=drive_root,
        reports=reports,
        external_blockers=external_blockers,
    )

    write_json(output_dir / "dataset_inventory.json", dataset_inventory)
    write_json(output_dir / "calibration_report.json", calibration_report)
    write_json(output_dir / "events_pipeline_report.json", events_pipeline_report)
    write_json(output_dir / "contract_report.json", group1_contract_report)
    write_json(output_dir / "external_blockers.json", external_blockers)
    write_json(output_dir / "phase0_readiness.json", readiness)

    print(f"Wrote Phase 0 audit evidence to {output_dir}")
    print(
        "Phase 0 status: "
        f"{readiness['phase0_status']} "
        f"(internal={readiness['internal_status']}, external={readiness['external_status']})"
    )

    if args.fail_on_internal_errors and readiness["internal_status"] != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

