# Training Hyperparameter Playbook

Reference for which Ultralytics training knobs we've already evaluated, which are worth tuning next, and the order to try them in. Operational runbook is [PHASE1_HANDOFF.md](PHASE1_HANDOFF.md); this doc is the "what knob and why" companion. Priority groupings below (Priority 1–4) are tuning *order* — not the same thing as the customer-readiness Tiers in [PRODUCT_VISION.md](PRODUCT_VISION.md).

**Default values shown are Ultralytics defaults.** Current joint v2 values from [configs/yolo/experiments_joint_v2.yaml](../configs/yolo/experiments_joint_v2.yaml). New augmentation/optimization knobs must be wired through both [scripts/03_train_yolov8_seg.py](../scripts/03_train_yolov8_seg.py) argparse and [scripts/03b_train_experiment_matrix.py](../scripts/03b_train_experiment_matrix.py) extraction — they are **not** auto-forwarded from YAML.

---

## Already evaluated — locked in for joint v2

| Knob | Value | Why this value |
|---|---|---|
| `optimizer` | `SGD` | `optimizer: auto` silently picks AdamW(lr=0.002) and collapsed Box mAP50 0.696 → 0.004 in one epoch on this warm start. SGD preserves it. |
| `lr0` | `0.001` | Peak LR with `cos_lr: true` + `warmup_epochs: 3.0` ramping in. Higher values destroy the warm start. |
| `cos_lr` | `true` | Smooth decay to near-zero by epoch 100. |
| `auto_augment` | `null` | Default `randaugment` produced starburst train tiles on 160 px Duke chips and tanked `joint_v2_duke160_naip_v11s_e1002` to 0.247. |
| `erasing` | `0.0` | Default `0.4` destroys small Duke arrays during training. |
| `translate` | `0.0` | Default `0.1` = 16 px shift on 160 px Duke chips, comparable to small-array diameter. |
| `mosaic` | `0.5` | Half the default. Useful for small-object diversity, but full strength dominated gradients in joint v1. |
| `close_mosaic` | `5` | Disable mosaic in last 5 epochs for clean fine-tuning. |
| `copy_paste` | `0.0` | Off; harmful for aerial imagery. |
| `perspective` | `0.0` | Orthophotos shouldn't have perspective warp. |
| `degrees` | `5.0` | Mild rotation. Safe for aerial. |
| `shear` | `2.0` | Mild. Safe. |

---

## Most promising untried knobs (in priority order)

### Priority 1 — strongest warm-start preservation

#### `freeze: 10`
**Default:** `0` (no freezing).
**What it does:** Freezes the first N layers (the backbone). Only the neck + head update.
**When to try:** Final NAIP test mAP50 lands in the 0.45–0.55 band. This is the **strongest preservation move available** — try this before any aug tweak, because aug-only fixes can't help if the backbone is being walked away from the warm start.
**How to apply:** Need to wire `--freeze` into [scripts/03_train_yolov8_seg.py](../scripts/03_train_yolov8_seg.py) argparse and matrix runner extraction (currently neither passes it). Add to YAML as `freeze: 10`.

#### `flipud: 0.5`
**Default:** `0.0` (off).
**What it does:** Vertical flip with 50% probability.
**Why it's high-EV:** Aerial imagery has no "up". Solar arrays are rotation-invariant by class definition. This is free 2× data variety with zero downside on orthophotos.
**Caveat:** Confirm Duke labels are stored as polygons (they are — see [.claude/rules/data-pipeline.md](../.claude/rules/data-pipeline.md)) so vertical flip transforms them correctly. YOLO handles polygon flips natively.

### Priority 2 — Duke-chip color stress

#### `hsv_s: 0.4` and `hsv_v: 0.2`
**Defaults:** `0.7` and `0.4` (currently active — Ultralytics defaults).
**What they do:** Saturation and value/brightness jitter. Heavy on 160 px Duke chips when upscaled 4× to 640.
**When to try:** Duke side under-performs (Duke test mAP50 < 0.10 with NAIP test ≥ 0.55). Tighten both together — saturation alone won't help if value is also stressing the model.
**Note:** `hsv_h: 0.015` is already low; leave alone.

### Priority 3 — geometric tightening

#### `scale: 0.3`
**Default:** `0.5` (currently set, also Ultralytics default).
**What it does:** Random scaling factor. `0.5` = random 0.5×–1.5× scale per image.
**When to try:** NAIP test 0.45–0.55 after Priority 1. The one knob Copilot's analysis got right — too aggressive for a warm-start scenario, though not the primary suspect.

### Priority 4 — fine-tuning profile tightening

#### `lrf: 0.001` + `warmup_epochs: 5`
**Defaults:** `0.01` and `3.0`.
**What they do:** `lrf` is the final LR as a fraction of `lr0` (so final LR = `lr0 * lrf` = 1e-6 instead of the current 1e-5). `warmup_epochs` extends the ramp from 3 to 5 epochs.
**When to try:** If results are close but the per-epoch curve oscillates near the end. Reduces final-stage step size and gives the warm start more buffer at the start.

#### `weight_decay: 0.001`
**Default:** `0.0005`.
**What it does:** L2 regularization strength. 2× stronger.
**When to try:** Symptoms of overfitting on the 29× oversampled NAIP tiles — train loss continuing to drop while val plateaus.

#### `patience: 50`
**Default:** `100` (Ultralytics) / `20` (our scripts default).
**What it does:** Epochs without improvement before early stop.
**When to try:** Loss curve is slow but improving. Warm-start runs sometimes need more patience than cold-start.

---

## Knobs not worth touching

| Knob | Why skip |
|---|---|
| `mixup` (default 0.0) | Generally hurts detection. |
| `bgr` (default 0.0) | Channel swap; we already have RGB. |
| `cache` (default False) | Speed-only; doesn't affect quality. |
| `rect` (default False) | Only matters for non-square aspect ratios. |
| `single_cls` | Implicit — we have one class. |
| `box` / `cls` / `dfl` loss weights | Defaults are fine for single-class detection. Tuning is research-grade with diminishing returns vs. data work. |
| `momentum` (default 0.937) | Standard SGD momentum. Don't change without a reason. |

---

## Capacity / resolution — the big lever after joint v2

Once joint v2 produces a stable checkpoint, the next move is capacity, not aug.

| Knob | Current | Switch to | Effect |
|---|---|---|---|
| `model` | `models/sahi_baseline_train7.pt` (yolo11s-seg) | `yolo11m-seg.pt` | ~3× capacity. Cold-start (no v11m baseline checkpoint exists). Defer until R0 lands cleanly; m-size is for the ≥0.60 NAIP / ≥0.20 Duke band only. |
| `imgsz` | `640` (preset `small`) | `768` (preset `medium`) | Duke chips get ~5× upscale instead of 4× — small arrays get more pixels. Training ~1.5× slower. |

---

## Recommended ordering by NAIP test mAP50 result

After each overnight, read the metrics and pick **one** knob per next run. Don't change multiple variables at once.

| Result band | Read | Next move |
|---|---|---|
| **NAIP test < 0.45** | Something else is broken, not just augmentation | Don't tune knobs. Re-investigate: check `args.yaml`, train_batch images, dataset cache, warm-start checkpoint integrity. |
| **NAIP test 0.45–0.55** | Warm start partially eroded | Try in this order, one per overnight: (1) `freeze: 10`, (2) `scale: 0.3`, (3) `hsv_s: 0.4` + `hsv_v: 0.2`, (4) `flipud: 0.5`, (5) `lrf: 0.001` + `warmup_epochs: 5` |
| **NAIP test ≥ 0.56, Duke test < 0.10** | NAIP held, Duke ignored | Not an aug problem — Duke 4× upscale is the bottleneck. Try `imgsz: 768` or smaller Duke fine-tune subset. |
| **NAIP test ≥ 0.56, Duke test ≥ 0.10** | Both domains learning | No aug changes. Run threshold sweep, ship as new production checkpoint, queue v11m scale-up. |
| **NAIP test ≥ 0.70** | GA gate cleared | Stop tuning. Promote checkpoint, calibrate, update Phase 1 status in [QUARTER_PLAN_OVERVIEW.md](QUARTER_PLAN_OVERVIEW.md). |

---

## Adding a new knob to the pipeline

If a future run needs a knob that isn't already wired through:

1. Add to YAML config under the experiment block.
2. Add `--<knob>` flag to argparse in [scripts/03_train_yolov8_seg.py](../scripts/03_train_yolov8_seg.py) (around line 41–50).
3. Add to `train_kwargs` block in the same script (around line 105–125), guarding with `if args.<knob> is not None:` so YAML omission falls back to the Ultralytics default.
4. Add `<knob> = exp.get("<knob>")` extraction in [scripts/03b_train_experiment_matrix.py](../scripts/03b_train_experiment_matrix.py) (around line 83–95).
5. Add the corresponding `*(["--<knob>", str(<knob>)] if <knob> is not None else [])` to the `train_cmd` list (around line 105–115).

For knobs where YAML `null` should mean "force-disable" (different from "key absent"), use the `__missing__` sentinel pattern that `auto_augment` uses.
