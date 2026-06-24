# Dataset Splits

This folder is for small, version-controlled split definitions only.

Use it for files that define exactly which samples belong to a benchmark,
training, validation, or test split. Examples:

```text
coco17_val2017_first100.txt
cricket_train.json
cricket_val.json
cricket_test.json
```

Do not put images, videos, labels, predictions, or generated benchmark payloads
here. Raw datasets belong under `data/raw/` and local generated data belongs
under `data/derived/`.
