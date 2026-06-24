# data/

Local dataset storage. Keep raw inputs immutable; write derived artifacts elsewhere.

```text
data/
  raw/        # downloaded datasets, e.g. coco/{val2017,annotations}   (LOCAL, ignored)
  derived/    # frames, predictions_2d, triangulated_3d, ...           (LOCAL, ignored)
  splits/     # small, stable split definitions (IDs only, no media)   (TRACKED)
```

Only `data/splits/` and this `README.md` are committed. Split files should carry stable
IDs for `camera_id`, `timestamp_ns`, `frame_id`, `player_id`, `event_type`,
`calibration_id`, and `occlusion_tags`.

Get the first benchmark dataset (COCO 2017 val keypoints):

```bash
python3 scripts/download_coco_keypoints.py --remove-archives
```

Normalize model outputs to `coco_17` via `configs/keypoint_mappings.yaml` before
comparing. See [docs/datasets.md](../docs/datasets.md).
