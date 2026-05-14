.PHONY: help eval-production eval-r2 relabel-overlays train-r0 demo-aoi train-soiling score-aoi sahi-combined

# Default weights for eval targets (override: make eval-production WEIGHTS=models/my.pt)
WEIGHTS ?= production
RUN_NAME ?= production_eval

help:
	@echo ""
	@echo "solar-soiling-ml — common targets"
	@echo ""
	@echo "  eval-production       SAHI F1 sweep on production weights (05d) — the number that matters"
	@echo "  eval-r2               SAHI F1 sweep on R2 weights (r2_cameron_20260509)"
	@echo "  sahi-combined         Re-run 05d with perform_standard_pred=true (combined full+slice)"
	@echo "  relabel-overlays      Render FP/FN overlays for next Roboflow labeling batch (05c + 18)"
	@echo "  train-r0              Retrain R0 on current NAIP labels (warm-start from SAHI baseline)"
	@echo "  demo-aoi              Run full pipeline on Santa Cruz test AOI (solarsoiled run)"
	@echo "  train-soiling         Retrain Stage 2 soiling model (script 10)"
	@echo "  score-aoi             Score detected arrays for soiling risk (script 11)"
	@echo ""
	@echo "  WEIGHTS override:     make eval-production WEIGHTS=models/r2_cameron_20260509.pt"
	@echo ""

# ── Stage 1 eval ─────────────────────────────────────────────────────────────

eval-production:
	PYTHONPATH=. python scripts/05d_sahi_threshold_sweep.py \
		--weights $(WEIGHTS) \
		--config configs/yolo/thresholds_sahi.yaml \
		--run-name production_eval

eval-r2:
	PYTHONPATH=. python scripts/05d_sahi_threshold_sweep.py \
		--weights models/r2_cameron_20260509.pt \
		--config configs/yolo/thresholds_sahi.yaml \
		--run-name r2_cameron_20260509_reeval

# Re-run eval with SAHI combined mode (full-tile + slice merge).
# Edit configs/yolo/thresholds_sahi_combined.yaml to set perform_standard_pred: true,
# or flip it manually before running.
sahi-combined:
	@if ! grep -q 'perform_standard_pred: true' configs/yolo/thresholds_sahi.yaml; then \
		echo ""; \
		echo "WARNING: perform_standard_pred is not 'true' in configs/yolo/thresholds_sahi.yaml"; \
		echo "Edit the config first, then re-run this target."; \
		echo ""; \
		exit 1; \
	fi
	PYTHONPATH=. python scripts/05d_sahi_threshold_sweep.py \
		--weights $(WEIGHTS) \
		--config configs/yolo/thresholds_sahi.yaml \
		--run-name $(RUN_NAME)_combined

# ── Label audit ───────────────────────────────────────────────────────────────

relabel-overlays:
	PYTHONPATH=. python scripts/05c_per_detection_rca.py \
		--weights $(WEIGHTS) \
		--data data/yolo/naip/data.yaml \
		--splits val test \
		--sahi --conf 0.05 --iou 0.50 \
		--run-name $(RUN_NAME)
	PYTHONPATH=. python scripts/labeling/18_bucket_overlays.py \
		--csv outputs/eval/$(RUN_NAME)/per_detection.csv \
		--bucket confident_fp --top 20
	PYTHONPATH=. python scripts/labeling/18_bucket_overlays.py \
		--csv outputs/eval/$(RUN_NAME)/per_detection.csv \
		--bucket worst_small_fn --top 20
	@echo ""
	@echo "Overlays written to outputs/label_viz/$(RUN_NAME)/"

# ── Training ──────────────────────────────────────────────────────────────────

train-r0:
	PYTHONPATH=. python scripts/03b_train_experiment_matrix.py \
		--config configs/yolo/experiments_joint_v2_ramp.yaml \
		--experiment R0 \
		--data data/yolo/naip/data.yaml

# ── End-to-end demo ───────────────────────────────────────────────────────────

# Santa Cruz test AOI (small bbox for smoke test)
AOI ?= -122.05,36.97,-121.98,37.03
PARTNER ?= smoketest
LAST_CLEANED ?= 2025-12-01
SOILING_MODEL ?= soiling_production

demo-aoi:
	solarsoiled run \
		--aoi "$(AOI)" \
		--weights production \
		--soiling-model $(SOILING_MODEL) \
		--last-cleaned $(LAST_CLEANED) \
		--partner-id $(PARTNER)

# ── Stage 2 soiling ───────────────────────────────────────────────────────────

train-soiling:
	PYTHONPATH=. python scripts/10_train_soiling_model.py --run-name run_latest

score-aoi:
	PYTHONPATH=. python scripts/11_predict_soiling_risk.py \
		--model runs/soiling/run_latest/model.ubj
