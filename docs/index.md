# Documentation

This workspace benchmarks pose estimation models so a team can compare them on the
same datasets, with the same metrics, and merge results cleanly into `main`.

If you only read one page, read **[getting-started.md](getting-started.md)**.

## Read these in order (Start here)

| # | Doc | What you get |
| - | --- | ------------ |
| 1 | [getting-started.md](getting-started.md) | Install prerequisites and run your first benchmark end to end, with a success check after every step. |
| 2 | [concepts.md](concepts.md) | The mental model: tracked vs. local, what a "run" is, the five-stage pipeline, and why it's built this way. |
| 3 | [workflow.md](workflow.md) | The `prepare → smoke → run → aggregate → report` pipeline in depth — what each stage does and when to use it. |
| 4 | [collaboration.md](collaboration.md) | The single source of truth for what goes to GitHub vs. stays local, plus the multi-person branch/merge workflow. |

## Look these up when you need them (Reference)

| Doc | Use it when… |
| --- | ------------ |
| [improving-models.md](improving-models.md) | …you're reducing noise, smoothing, fusing cameras, or fine-tuning. This is the core project work. |
| [scripts.md](scripts.md) | …you want to know exactly what a script does, its inputs/outputs, and its key flags. |
| [configuration.md](configuration.md) | …you need to read or edit a `configs/*.yaml` file. |
| [models.md](models.md) | …you're choosing a model or hit a model-specific quirk (OpenPose, Sapiens2, ViTPose). |
| [datasets.md](datasets.md) | …you need to download or understand a dataset. |
| [metrics.md](metrics.md) | …you want to know what a metric means or which ones are reported. |
| [results-format.md](results-format.md) | …you're inspecting a run folder or wondering what's safe to commit. |
| [adding-a-model.md](adding-a-model.md) | …you want to benchmark a model that isn't in the registry yet. |
| [troubleshooting.md](troubleshooting.md) | …something failed (downloads, OpenPose build, CUDA, empty benchmarks). |

## Common commands

```bash
# 1. one-time: install model envs + download COCO
python3 scripts/benchmark.py prepare --models all --datasets coco17_val2017

# 2. confirm assets are present
python3 scripts/check_assets.py --models all --fail-missing

# 3. quick readiness check
python3 scripts/benchmark.py smoke --models all

# 4. a real benchmark (benchmark-ready: yolo26x_pose, rtmw_l, rtmw_x, rtmpose_l_wholebody)
python3 scripts/benchmark.py run --models yolo26x_pose --datasets coco17_val2017

# 5. preview the comparison locally (CI does this for real on main)
python3 scripts/benchmark.py aggregate
python3 scripts/benchmark.py report
```
