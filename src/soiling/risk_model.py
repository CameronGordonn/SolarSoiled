"""XGBoost training + prediction wrappers with spatial cross-validation.

Soiling is strongly spatially autocorrelated — a random KFold would leak
neighboring stations between folds and inflate metrics. We instead cluster
stations by coordinate and hold out whole clusters.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SpatialCVConfig:
    n_folds: int = 5
    cluster_km: float = 50.0
    random_state: int = 42


def _feature_medians(X: pd.DataFrame) -> dict[str, float]:
    med = X.median(numeric_only=True).fillna(0.0)
    return {str(k): float(v) for k, v in med.items()}


def impute_with_feature_medians(X: pd.DataFrame, medians: Mapping[str, float] | None) -> pd.DataFrame:
    """Fill missing values with medians learned on the training split.

    XGBoost can handle NaNs directly, but we impute so the saved model sees the
    same value scale at CV, holdout, and inference time.
    """
    if not medians:
        return X.copy()

    X_filled = X.copy()
    fill_values = {c: float(medians[c]) for c in X_filled.columns if c in medians}
    if fill_values:
        X_filled = X_filled.fillna(value=fill_values)

    remaining = X_filled.columns[X_filled.isna().any()]
    if len(remaining) > 0:
        fallback = {}
        for col in remaining:
            value = X_filled[col].median(skipna=True)
            fallback[col] = 0.0 if pd.isna(value) else float(value)
        X_filled = X_filled.fillna(value=fallback)
    return X_filled


def _spatial_clusters(lats: np.ndarray, lons: np.ndarray, cluster_km: float) -> np.ndarray:
    """Greedy lat/lon bucketing → integer cluster labels. Good enough for v1."""
    # 1 degree latitude ≈ 111 km. Longitude varies with latitude; approximate
    # with mean-latitude cosine correction.
    deg_per_km_lat = 1.0 / 111.0
    mean_lat = float(np.nanmean(lats))
    deg_per_km_lon = 1.0 / (111.0 * max(0.1, np.cos(np.radians(mean_lat))))
    lat_bin = np.floor(lats * (deg_per_km_lat * cluster_km) ** -1).astype(int)
    lon_bin = np.floor(lons * (deg_per_km_lon * cluster_km) ** -1).astype(int)
    combined = lat_bin * 10_000 + lon_bin
    _, inverse = np.unique(combined, return_inverse=True)
    return inverse


def train_risk_model(
    X: pd.DataFrame,
    y: np.ndarray,
    lats: Optional[np.ndarray] = None,
    lons: Optional[np.ndarray] = None,
    xgb_params: Optional[dict] = None,
    cv: Optional[SpatialCVConfig] = None,
    target_mode: str = "binary",
    iwsr_target: Optional[np.ndarray] = None,
    sample_weight: Optional[np.ndarray] = None,
    iwsr_risk_threshold: float = 0.97,
):
    """Train an XGBoost soiling model with optional spatial CV.

    `target_mode`:
      - "binary"     : fit XGBClassifier on y (0/1); report AUC/AP from predict_proba.
      - "regression" : fit XGBRegressor on `iwsr_target` (continuous); derive AUC by
                       thresholding predictions at `iwsr_risk_threshold`, and report
                       Spearman rank correlation against the true IWSR.

    `sample_weight` is forwarded to every fold and the final full-data fit.

    Returns (fitted_model_on_all_data, cv_metrics_dict).
    """
    try:
        import xgboost as xgb
        from sklearn.model_selection import GroupKFold
        from sklearn.metrics import roc_auc_score, average_precision_score
        from scipy.stats import spearmanr
    except ImportError as err:
        raise ImportError("xgboost, scikit-learn, and scipy are required") from err

    if target_mode not in {"binary", "regression"}:
        raise ValueError(f"target_mode must be 'binary' or 'regression', got {target_mode!r}")
    if target_mode == "regression" and iwsr_target is None:
        raise ValueError("target_mode='regression' requires iwsr_target")

    base = {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }
    if xgb_params:
        base.update({k: v for k, v in xgb_params.items() if k != "objective"})

    if target_mode == "binary":
        params = {**base, "objective": "binary:logistic", "eval_metric": "auc"}
        def _make(): return xgb.XGBClassifier(**params)
        fit_target = y
    else:
        params = {**base, "objective": "reg:squarederror"}
        def _make(): return xgb.XGBRegressor(**params)
        fit_target = np.asarray(iwsr_target, dtype=float)

    def _predict_score(model, X_te):
        if target_mode == "binary":
            return model.predict_proba(X_te)[:, 1]
        # Regression: higher IWSR = cleaner. For at-risk ranking we flip sign.
        return -model.predict(X_te)

    cv_metrics: dict = {}
    oof_scores = np.full(len(X), np.nan)
    if cv is not None and lats is not None and lons is not None:
        clusters = _spatial_clusters(np.asarray(lats), np.asarray(lons), cv.cluster_km)
        n_unique = len(np.unique(clusters))
        n_folds = min(cv.n_folds, n_unique)
        if n_folds < 2:
            logger.warning("Only %d spatial clusters — skipping CV", n_unique)
        else:
            gkf = GroupKFold(n_splits=n_folds)
            aucs, aps, spearmans = [], [], []
            for fold, (tr, te) in enumerate(gkf.split(X, y, groups=clusters)):
                m = _make()
                medians = _feature_medians(X.iloc[tr])
                X_tr = impute_with_feature_medians(X.iloc[tr], medians)
                X_te = impute_with_feature_medians(X.iloc[te], medians)
                fit_kw = {}
                if sample_weight is not None:
                    fit_kw["sample_weight"] = np.asarray(sample_weight)[tr]
                m.fit(X_tr, fit_target[tr], **fit_kw)
                p = _predict_score(m, X_te)
                oof_scores[te] = p
                if len(np.unique(y[te])) > 1:
                    aucs.append(roc_auc_score(y[te], p))
                    aps.append(average_precision_score(y[te], p))
                else:
                    logger.warning("Fold %d has single-class test set — skipping AUC", fold)
                if target_mode == "regression" and iwsr_target is not None:
                    rho, _ = spearmanr(iwsr_target[te], m.predict(X_te))
                    if rho == rho:  # not NaN
                        spearmans.append(float(rho))
            cv_metrics = {
                "target_mode": target_mode,
                "n_folds": n_folds,
                "mean_auc": float(np.mean(aucs)) if aucs else float("nan"),
                "mean_ap": float(np.mean(aps)) if aps else float("nan"),
                "fold_aucs": [float(a) for a in aucs],
            }
            if spearmans:
                cv_metrics["mean_spearman"] = float(np.mean(spearmans))
                cv_metrics["fold_spearmans"] = spearmans

    calibrator = None
    if cv_metrics and not np.isnan(oof_scores).all():
        valid = ~np.isnan(oof_scores)
        if len(np.unique(y[valid])) > 1:
            try:
                calibrator = fit_isotonic_calibrator(oof_scores[valid], y[valid].astype(float))
            except Exception as exc:
                logger.warning("Calibrator fit failed: %s", exc)

    model = _make()
    feature_medians = _feature_medians(X)
    X_final = impute_with_feature_medians(X, feature_medians)
    fit_kw = {}
    if sample_weight is not None:
        fit_kw["sample_weight"] = np.asarray(sample_weight)
    model.fit(X_final, fit_target, **fit_kw)
    model.feature_medians_ = feature_medians
    return model, cv_metrics, calibrator


def save_model(model, out_dir: Path, feature_names: Sequence[str], metrics: dict, calibrator=None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.ubj"
    model.save_model(str(model_path))
    (out_dir / "feature_names.json").write_text(json.dumps(list(feature_names), indent=2))
    feature_medians = getattr(model, "feature_medians_", None)
    if feature_medians is not None:
        (out_dir / "feature_medians.json").write_text(json.dumps(feature_medians, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if calibrator is not None:
        try:
            import joblib
            joblib.dump(calibrator, out_dir / "calibrator.joblib")
        except Exception as exc:
            logger.warning("Failed to persist calibrator: %s", exc)
    return model_path


def fit_isotonic_calibrator(scores: np.ndarray, y: np.ndarray):
    """Fit an isotonic regression mapping raw classifier/regressor scores to
    probabilities. Use out-of-fold predictions for `scores` to avoid optimism.
    """
    from sklearn.isotonic import IsotonicRegression

    cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    cal.fit(scores, y)
    return cal


def load_calibrator(model_dir: Path):
    cal_path = Path(model_dir) / "calibrator.joblib"
    if not cal_path.exists():
        return None
    try:
        import joblib
        return joblib.load(cal_path)
    except Exception as exc:
        logger.warning("Failed to load calibrator: %s", exc)
        return None


def load_model(model_path: Path):
    import xgboost as xgb

    metrics_path = Path(model_path).parent / "metrics.json"
    target_mode = "binary"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        target_mode = metrics.get("target_mode", "binary")

    model = xgb.XGBRegressor() if target_mode == "regression" else xgb.XGBClassifier()
    model.load_model(str(model_path))
    feat_path = Path(model_path).parent / "feature_names.json"
    feature_names = json.loads(feat_path.read_text()) if feat_path.exists() else None
    return model, feature_names


def load_feature_medians(model_dir: Path):
    med_path = Path(model_dir) / "feature_medians.json"
    if not med_path.exists():
        return None
    return json.loads(med_path.read_text())


def predict_risk(model, X: pd.DataFrame, iwsr_risk_threshold: float = 0.97, calibrator=None) -> np.ndarray:
    """Return a per-row risk score in [0, 1]: higher = more at-risk.

    Binary classifier: returns predict_proba of positive class directly.
    Regressor: returns -predict (higher score = more at-risk for ranking).

    If `calibrator` is provided (from `fit_isotonic_calibrator`), the raw score
    is mapped to a calibrated [0, 1] probability that matches the empirical
    positive rate seen during CV — required for an interpretable risk_score.
    """
    import xgboost as xgb

    if isinstance(model, xgb.XGBClassifier):
        raw = model.predict_proba(X)[:, 1]
    else:
        raw = -model.predict(X)
    if calibrator is not None:
        return np.clip(calibrator.predict(raw), 0.0, 1.0)
    if isinstance(model, xgb.XGBClassifier):
        return raw
    # Without calibrator, normalize the regression score to [0, 1] via threshold.
    iwsr_pred = -raw
    score = (iwsr_risk_threshold - iwsr_pred) / iwsr_risk_threshold
    return np.clip(score, 0.0, 1.0)
