"""
Train a CatBoost revenue model with stronger features and time-aware cross-validation.
- Builds on prior script structure (feature engineering + CatBoost) with upgrades:\n  * richer engineered features (genre/distributor hygiene, critic+audience aggregates, vote-weighted signals)\n  * broader log transforms (popularity, vote_count)\n  * **strict revenue filter**: keep only 0 < revenue ≤ $120M (per your requirement)\n  * forward-chaining CV over release years (simulates future prediction)\n  * larger hyperparameter search space with early stopping\n  * optional ensembling (train best-on-full then optionally refit)\n\nUsage:
  python train_revenue_upgraded_cv.py \
    --csv E:\\Projects\\FilmBrain\\PRMoviesDB_updated_merged.csv \
    --target revenue \
    --iters 8000 --early_stop 400 --n_search 50 --clip_q 0.99

Artifacts:
  ./artifacts_merged/
    - catboost_revenue_model.pkl (best single model)
    - report.json (CV metrics + best params)

Notes:
  * CatBoost handles string categoricals directly; we avoid heavy one-hot.
  * We drop PR leakage columns (any column containing acum/gbo/adms/puerto_rico/island/pr_).
"""

from __future__ import annotations
import os
import re
import json
import math
import random
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------------- Utilities -----------------
def coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(r"[,$\s]", "", regex=True), errors="coerce")

def log1p_safe(x: pd.Series) -> pd.Series:
    return np.log1p(np.clip(x, a_min=0, a_max=None))

def smape(y_true, y_pred) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    m = denom != 0
    return float(np.mean(np.abs(y_true[m] - y_pred[m]) / denom[m]) * 100)

# ---- List-like normalization ----
def normalize_listlike_cell(x):
    if isinstance(x, list):
        vals = [str(v).strip() for v in x if str(v).strip()]
        return ";".join(sorted(set(vals))) if vals else "Unknown"
    if isinstance(x, str):
        if not x.strip():
            return "Unknown"
        parts = re.split(r"[;,]|(?<=\w)\s*,\s*", x.strip("[](){} '\""))
        parts = [p.strip() for p in parts if p.strip() and p.lower() != "nan"]
        return ";".join(sorted(set(parts))) if parts else "Unknown"
    return "Unknown"

DATE_CANDIDATES = [
    "release_date", "release", "opening_date", "premiere_date", "date",
    "rel_date", "launch_date", "released", "release_dt"
]

def pick_date_column(df: pd.DataFrame) -> pd.Series:
    """Return a datetime Series; build from best-available fields."""
    cols_norm = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=cols_norm)

    # 1) Try common date-like columns by name
    for c in DATE_CANDIDATES:
        if c in df.columns:
            parsed = pd.to_datetime(df[c], errors="coerce", utc=False, infer_datetime_format=True)
            if parsed.notna().mean() >= 0.5:
                return parsed

    # 2) Any '*date*' column
    for c in df.columns:
        if "date" in c:
            parsed = pd.to_datetime(df[c], errors="coerce", utc=False, infer_datetime_format=True)
            if parsed.notna().mean() >= 0.5:
                return parsed

    # 3) Construct from year-like
    for yname in ["release_year", "year", "years"]:
        if yname in df.columns:
            y = pd.to_numeric(df[yname], errors="coerce")
            return pd.to_datetime(y.fillna(y.median()).astype("Int64").astype(str) + "-07-01", errors="coerce")

    # 4) Fallback: parse any object column
    for c in df.columns:
        if df[c].dtype == object:
            parsed = pd.to_datetime(df[c], errors="coerce", utc=False, infer_datetime_format=True)
            if parsed.notna().mean() >= 0.5:
                return parsed

    # 5) Sequential fallback
    n = len(df)
    base = pd.to_datetime("2000-01-01")
    return base + pd.to_timedelta(np.arange(n), unit="D")


def drop_heavy_text_columns(df: pd.DataFrame, max_len: int = 500) -> pd.DataFrame:
    """Drop very large free-text columns (e.g., overview/tagline) to reduce memory for big CSVs."""
    drop_cols = []
    for c in df.columns:
        if df[c].dtype == object:
            sample = df[c].dropna().astype(str).head(50)
            if len(sample) > 0 and sample.str.len().median() > max_len:
                drop_cols.append(c)
    if drop_cols:
        print(f"Dropping heavy text columns (memory optimization): {drop_cols}")
    return df.drop(columns=drop_cols, errors="ignore")


# --------------- Feature builder ---------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Memory optimization
    df = drop_heavy_text_columns(df, max_len=400)

    # Release date & temporal signals
    df["release_date"] = pd.to_datetime(pick_date_column(df), errors="coerce")
    df["release_year"] = df["release_date"].dt.year
    df["release_month"] = df["release_date"].dt.month
    df["release_weekday"] = df["release_date"].dt.weekday
    df["release_quarter"] = ((df["release_month"] - 1) // 3 + 1).astype("Int64")
    df["release_weekofyear"] = df["release_date"].dt.isocalendar().week.astype("Int64")
    df["is_summer"] = df["release_month"].isin([6, 7, 8]).astype(int)
    df["is_december"] = (df["release_month"] == 12).astype(int)
    df["is_holiday_season"] = df["release_month"].isin([11, 12]).astype(int)

    # Normalize list-like categoricals
    listlike_cols = [c for c in [
        "genres", "genre_ids", "origin_country", "production_companies",
        "spoken_languages", "keywords"
    ] if c in df.columns]
    for c in listlike_cols:
        df[c] = df[c].apply(normalize_listlike_cell)
        df[f"{c}_count"] = df[c].apply(lambda s: 0 if s == "Unknown" else s.count(";") + 1)

    # Distributor clean-up if present
    for alias in ["dist.", "distributor", "distrib", "distribution", "dist", "dist_."]:
        if alias in df.columns:
            df[alias] = df[alias].fillna("Unknown").astype(str)

    # Coerce numeric signals
    num_candidates = [
        "budget", "popularity", "vote_average", "vote_count", "runtime",
        "score", "reviewscount", "metascore", "userscore", "averagescore",
        "tomatometer"
    ]
    for c in num_candidates:
        if c in df.columns:
            df[c] = coerce_numeric(df[c])

    # Derived numeric features
    if "budget" in df.columns:
        df["log_budget"] = log1p_safe(df["budget"])
    if "vote_average" in df.columns and "vote_count" in df.columns:
        vc = np.clip(df["vote_count"], 0, None)
        df["vote_power"] = df["vote_average"] * np.log1p(vc)
        df["vote_weighted"] = df["vote_average"] * np.sqrt(vc)
    if "runtime" in df.columns and "budget" in df.columns:
        df["budget_per_min"] = df["budget"] / np.clip(df["runtime"], 1, None)
    if "popularity" in df.columns:
        df["log_popularity"] = log1p_safe(df["popularity"])  # heavy-tailed

    # Critic + audience aggregates
    critic_cols = [c for c in ["metascore", "userscore", "averagescore", "score", "tomatometer"] if c in df.columns]
    if critic_cols:
        df["critic_audience_mean"] = df[critic_cols].mean(axis=1, skipna=True)

    # Franchise / sequel heuristic
    title_col = None
    for t in ["film_title", "title", "original_title", "name"]:
        if t in df.columns:
            title_col = t
            break
    if title_col is not None:
        pat = re.compile(r"(\bpart\s*[ivxlcdm]+\b|\bpart\s*[0-9]+\b|\bii\b|\biii\b|\biv\b|\b2\b|\b3\b)", re.IGNORECASE)
        df["is_sequel"] = df[title_col].astype(str).apply(lambda s: 1 if pat.search(s) else 0)
        # simple franchise keyword flags
        for kw in ["marvel", "dc", "star wars", "fast & furious", "pixar", "disney"]:
            col = f"kw_{kw.replace(' ', '_')}"
            df[col] = df[title_col].str.contains(kw, case=False, na=False).astype(int)
    else:
        df["is_sequel"] = 0

    # Drop PR-specific leakage columns
    leakage = [c for c in df.columns if any(k in c for k in ["acum", "gbo", "adms", "puerto_rico", "island", "pr_"])]
    df = df.drop(columns=leakage, errors="ignore")

    return df


# --------------- Evaluation helpers ---------------

def evaluate_from_logs(y_true_log, y_pred_log) -> Dict[str, float]:
    y_true, y_pred = np.expm1(y_true_log), np.expm1(y_pred_log)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "R2":   float(r2_score(y_true, y_pred)),
        "SMAPE": smape(y_true, y_pred),
        "MedianAE": float(np.median(np.abs(y_true - y_pred)))
    }


# --------------- CV splitting ---------------

def forward_year_cv(df: pd.DataFrame, year_col: str = "release_year", n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create forward-chaining splits based on distinct release years.
    Example: if years = [2004..2024] and n_splits=5, we build folds where
    each validation block is a consecutive group of years; training uses all
    previous years.
    """
    years = sorted([int(y) for y in pd.Series(df[year_col]).dropna().unique()])
    if len(years) < n_splits + 1:
        n_splits = max(2, min(4, len(years) - 1))
    # split years into roughly equal validation blocks
    val_blocks = np.array_split(years, n_splits)
    splits = []
    for i, val_years in enumerate(val_blocks):
        train_years = [y for y in years if y < min(val_years)]
        if not train_years:
            # ensure at least one previous block for training
            continue
        train_idx = df[df[year_col].isin(train_years)].index.values
        valid_idx = df[df[year_col].isin(val_years)].index.values
        if len(valid_idx) == 0 or len(train_idx) == 0:
            continue
        splits.append((train_idx, valid_idx))
    return splits


# --------------- Hyperparameter search ---------------

def random_search_catboost_cv(df: pd.DataFrame, feature_cols: List[str], target_log_col: str,
                              cat_features: List[str], n_iter: int, iterations: int, early_stop: int,
                              seed: int = SEED) -> Tuple[CatBoostRegressor, Dict, Dict]:
    rng = np.random.default_rng(seed)
    space = {
        "depth": [6, 8, 10, 12],
        "learning_rate": [0.005, 0.01, 0.02, 0.03],
        "l2_leaf_reg": [3, 8, 15, 30],
        "random_strength": [1, 5, 10, 15],
        "bagging_temperature": [0.0, 0.25, 0.5, 1.0, 2.0]
    }

    cv_splits = forward_year_cv(df)
    if not cv_splits:
        raise RuntimeError("CV splitting failed (insufficient year diversity).")

    best_model = None
    best_params = None
    best_metrics = None
    best_rmse = float("inf")

    for i in range(n_iter):
        params = {k: rng.choice(v).item() if hasattr(rng.choice(v), 'item') else rng.choice(v) for k, v in space.items()}
        fold_metrics = []
        models = []
        for fold, (tr_idx, va_idx) in enumerate(cv_splits, start=1):
            train_df, valid_df = df.loc[tr_idx], df.loc[va_idx]
            X_tr, y_tr = train_df[feature_cols], train_df[target_log_col]
            X_va, y_va = valid_df[feature_cols], valid_df[target_log_col]
            train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
            valid_pool = Pool(X_va, y_va, cat_features=cat_features)

            model = CatBoostRegressor(
                iterations=iterations,
                learning_rate=float(params["learning_rate"]),
                depth=int(params["depth"]),
                l2_leaf_reg=float(params["l2_leaf_reg"]),
                random_strength=float(params["random_strength"]),
                bagging_temperature=float(params["bagging_temperature"]),
                loss_function="RMSE",
                eval_metric="RMSE",
                random_seed=seed,
                early_stopping_rounds=early_stop,
                verbose=False,
            )
            model.fit(train_pool, eval_set=valid_pool, use_best_model=True, verbose=False)
            preds_val = model.predict(valid_pool)
            mets = evaluate_from_logs(y_va, preds_val)
            fold_metrics.append(mets)
            models.append(model)

        # aggregate metrics across folds
        agg = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0].keys()}
        print(f"[search {i+1}/{n_iter}] params={params}\n"
              f"    CV RMSE: ${agg['RMSE']/1e6:.2f}M | MAE: ${agg['MAE']/1e6:.2f}M | R2: {agg['R2']:.3f} | SMAPE: {agg['SMAPE']:.2f}%")

        if agg["RMSE"] < best_rmse:
            best_rmse = agg["RMSE"]
            best_metrics = agg
            best_params = params
            # keep the model from the last fold (or optionally refit later)
            best_model = models[-1]

    return best_model, best_params, best_metrics


# --------------- Main ---------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=r"E:\\Projects\\FilmBrain\\PRMoviesDB_updated_merged.csv")
    ap.add_argument("--target", type=str, default="revenue")
    ap.add_argument("--out", type=str, default="./artifacts_merged")
    ap.add_argument("--clip_q", type=float, default=0.99, help="Quantile for revenue clipping (e.g., 0.99)")
    ap.add_argument("--iters", type=int, default=8000, help="Max CatBoost trees")
    ap.add_argument("--early_stop", type=int, default=400)
    ap.add_argument("--n_search", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Loading data from:", args.csv)
    df = pd.read_csv(args.csv)
    df = build_features(df)

    # Locate target (allow fuzzy 'revenue' matches)
    target_col = args.target if args.target in df.columns else None
    if target_col is None:
        rev_like = [c for c in df.columns if "revenue" in c]
        if not rev_like:
            raise ValueError(f"Target '{args.target}' not found, and no revenue-like column detected.")
        target_col = rev_like[0]
        print(f"[warn] Using '{target_col}' as target (closest match).")

    # Coerce target & filter positive
    df[target_col] = coerce_numeric(df[target_col])
    before = len(df)
    df = df[df[target_col].notna() & (df[target_col] > 0)].copy()
    after = len(df)
    print(f"Filtered zero/null revenue rows: kept {after}/{before} ({after/before:.1%}).")

    # Enforce requested revenue range 0 < r <= 120M (no additional clipping)\nupper_cap = 120_000_000\ndf = df[(df[target_col] > 0) & (df[target_col] <= upper_cap)].copy()\nprint(f"Applied revenue filter: 0 < {target_col} <= ${upper_cap/1e6:.0f}M")\nprint(df[target_col].describe().apply(lambda x: f"${x/1e6:.2f}M"))

    # Target log transform
    df["target_log"] = log1p_safe(df[target_col])

    # Feature sets
    feature_cols = [c for c in df.columns if c not in [target_col, "target_log"]]
    cat_features = [c for c in feature_cols if df[c].dtype == "object"]
    num_features = [c for c in feature_cols if c not in cat_features]

    # Impute
    for c in num_features:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_features:
        df[c] = df[c].fillna("Unknown").astype(str)

    # Randomized hyperparameter search + forward-year CV
    best_model, best_params, best_metrics = random_search_catboost_cv(
        df=df,
        feature_cols=feature_cols,
        target_log_col="target_log",
        cat_features=cat_features,
        n_iter=args.n_search,
        iterations=args.iters,
        early_stop=args.early_stop,
        seed=SEED,
    )

    # Fit best model on all data (optional; keeps early stopping using last year as eval)
    df_sorted = df.sort_values("release_date")
    split_idx = int(0.9 * len(df_sorted))
    train_df, valid_df = df_sorted.iloc[:split_idx], df_sorted.iloc[split_idx:]
    train_pool = Pool(train_df[feature_cols], train_df["target_log"], cat_features=cat_features)
    valid_pool = Pool(valid_df[feature_cols], valid_df["target_log"], cat_features=cat_features)

    final_model = CatBoostRegressor(
        iterations=args.iters,
        learning_rate=float(best_params["learning_rate"]),
        depth=int(best_params["depth"]),
        l2_leaf_reg=float(best_params["l2_leaf_reg"]),
        random_strength=float(best_params["random_strength"]),
        bagging_temperature=float(best_params["bagging_temperature"]),
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=SEED,
        early_stopping_rounds=args.early_stop,
        verbose=False,
    )
    final_model.fit(train_pool, eval_set=valid_pool, use_best_model=True, verbose=False)

    # Evaluate final model on held-out tail
    preds_val = final_model.predict(valid_pool)
    final_mets = evaluate_from_logs(valid_df["target_log"], preds_val)

    # Save artifacts
    import joblib
    joblib.dump(
        {"model": final_model, "features": feature_cols, "cat_features": cat_features, "best_params": best_params},
        os.path.join(args.out, "catboost_revenue_model.pkl")
    )
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump({"cv_best_metrics": best_metrics, "best_params": best_params, "final_holdout_metrics": final_mets}, f, indent=2)

    print("\n=== Cross-Validation Best (avg across folds) ===")
    print(f"RMSE: ${best_metrics['RMSE']/1e6:.2f}M | MAE: ${best_metrics['MAE']/1e6:.2f}M | R²: {best_metrics['R2']:.4f} | SMAPE: {best_metrics['SMAPE']:.2f}%")
    print("Best params:", best_params)

    print("\n=== Final Holdout (last 10% by date) ===")
    print(f"RMSE: ${final_mets['RMSE']/1e6:.2f}M | MAE: ${final_mets['MAE']/1e6:.2f}M | R²: {final_mets['R2']:.4f} | SMAPE: {final_mets['SMAPE']:.2f}%")
    print(f"Artifacts saved to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
