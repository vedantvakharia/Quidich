#!/usr/bin/env python3
"""Write the Phase 1 provisional model shortlist for DS-001 perception."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


DEFAULT_CANDIDATES = [
    "yolo26x_pose",
    "rtmo_l",
    "rtmw_l",
    "rtmw_x",
    "rtmpose_l_wholebody",
    "dwpose_l_384",
    "sapiens2_1b_pose",
]

MODEL_NOTES = {
    "yolo26x_pose": {
        "protocol": "end_to_end",
        "person_bbox": True,
        "coco17_direct": True,
        "recommended_use": "first full-delivery baseline",
        "notes": "Detects person boxes and COCO-17 keypoints in one pass.",
    },
    "rtmo_l": {
        "protocol": "end_to_end",
        "person_bbox": True,
        "coco17_direct": True,
        "recommended_use": "second end-to-end candidate",
        "notes": "One-stage MMPose model; useful under crowded/occluded scenes if runtime is healthy.",
    },
    "rtmw_l": {
        "protocol": "top_down_with_detector_boxes",
        "person_bbox": False,
        "coco17_direct": False,
        "recommended_use": "whole-body comparison using YOLO boxes",
        "notes": "WholeBody-133 model; map to COCO-17 for Phase 1 contract.",
    },
    "rtmw_x": {
        "protocol": "top_down_with_detector_boxes",
        "person_bbox": False,
        "coco17_direct": False,
        "recommended_use": "heavier whole-body comparison using YOLO boxes",
        "notes": "Heaviest RTMW candidate; compare only after baseline path is stable.",
    },
    "rtmpose_l_wholebody": {
        "protocol": "top_down_with_detector_boxes",
        "person_bbox": False,
        "coco17_direct": False,
        "recommended_use": "top-down real-time baseline using YOLO boxes",
        "notes": "Good comparison point but not end-to-end without a detector.",
    },
    "dwpose_l_384": {
        "protocol": "top_down_or_detector_coupled",
        "person_bbox": False,
        "coco17_direct": False,
        "recommended_use": "whole-body detail check",
        "notes": "ONNX whole-body candidate; use for hands/feet detail after baseline.",
    },
    "sapiens2_1b_pose": {
        "protocol": "offline_teacher",
        "person_bbox": False,
        "coco17_direct": False,
        "recommended_use": "offline teacher / hard-frame analysis",
        "notes": "Not a first full-delivery production baseline.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default=str(ROOT / "configs" / "model_envs.yaml"))
    parser.add_argument("--run-id", default="p1-shortlist")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=str(ROOT / "drive" / "grp_1" / "phase_1" / "model_shortlist.md"))
    parser.add_argument("--models", nargs="+", default=DEFAULT_CANDIDATES)
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_shortlist(config: dict[str, Any], models: list[str], run_id: str) -> dict[str, Any]:
    rows = []
    for model_id in models:
        model = config.get("models", {}).get(model_id, {})
        notes = MODEL_NOTES.get(model_id, {})
        rows.append(
            {
                "model_id": model_id,
                "env_name": model.get("env_name"),
                "profile": model.get("profile"),
                "smoke_profile": model.get("smoke_profile"),
                "benchmark_runner": model.get("benchmark_runner"),
                "protocol": notes.get("protocol", "unknown"),
                "person_bbox": notes.get("person_bbox", False),
                "coco17_direct": notes.get("coco17_direct", False),
                "recommended_use": notes.get("recommended_use", ""),
                "notes": notes.get("notes", ""),
            }
        )
    return {
        "schema_version": "cricket_phase1_model_shortlist/v1",
        "run_id": run_id,
        "models": rows,
        "recommendation": {
            "first_full_delivery_baseline": "yolo26x_pose",
            "next_end_to_end_candidate": "rtmo_l",
            "top_down_comparison_group": ["rtmw_l", "rtmw_x", "rtmpose_l_wholebody", "dwpose_l_384"],
            "offline_teacher": "sapiens2_1b_pose",
        },
    }


def write_markdown(path: Path, shortlist: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 1 Model Shortlist",
        "",
        "This shortlist is provisional until DS-001 ground truth labels exist. It ranks models by integration shape and immediate usefulness for Phase 1.",
        "",
        "| Model | Env | Protocol | Bbox? | COCO-17 direct? | Recommended use |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in shortlist["models"]:
        lines.append(
            "| {model_id} | {env_name} | {protocol} | {bbox} | {coco} | {use} |".format(
                model_id=row["model_id"],
                env_name=row.get("env_name") or "",
                protocol=row.get("protocol") or "",
                bbox="yes" if row.get("person_bbox") else "no",
                coco="yes" if row.get("coco17_direct") else "no",
                use=row.get("recommended_use") or "",
            )
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "- Use `yolo26x_pose` for the first full-delivery DS-001 baseline.",
            "- Test `rtmo_l` next as the second end-to-end candidate.",
            "- Compare RTMW/RTMPose/DWPose only after detector-box plumbing is stable.",
            "- Keep Sapiens2 as an offline teacher/hard-frame analysis path.",
            "",
            "## Boundary",
            "",
            "This is not a final accuracy ranking. Final ranking requires manual DS-001/DS-002 labels.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_yaml(args.model_config)
    shortlist = build_shortlist(config, args.models, args.run_id)
    output_json = Path(args.output_json) if args.output_json else ROOT / "benchmarks" / "runs" / args.run_id / "p1_model_shortlist.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(shortlist, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(Path(args.output_md), shortlist)
    print(f"Wrote {output_json}")
    print(f"Wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

