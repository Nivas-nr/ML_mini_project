"""Score firms with the trained model.

Usage:
  python src/predict.py                       # demo: 10 random firms from the held-out test set
  python src/predict.py --csv my_firms.csv    # your own file with columns Attr1..Attr64
  python src/predict.py --csv my_firms.csv --out scored.csv
"""
import argparse
import json

import pandas as pd
from xgboost import XGBClassifier

from config import MODELS_DIR, RESULTS_DIR, SEED

MODEL_FILE = MODELS_DIR / "xgboost_model.json"
INFO_FILE = MODELS_DIR / "model_info.json"


def _rebuild(info: dict) -> XGBClassifier:
    """Fallback: retrain the final model (a few seconds) if the saved file is missing or unreadable."""
    from data import load_year, split
    from models import build_models

    print("Saved model could not be loaded; rebuilding it from the data (takes a few seconds)...")
    X, y = load_year(3)
    X_tr, _, y_tr, _ = split(X, y)
    model = build_models(float((y_tr == 0).sum() / (y_tr == 1).sum()))["XGBoost"].fit(X_tr, y_tr)
    MODELS_DIR.mkdir(exist_ok=True)
    model.save_model(MODEL_FILE)
    return model


def load_bundle() -> dict:
    if not INFO_FILE.exists():
        raise SystemExit("No trained model found. Run `python src/train.py` first.")
    info = json.load(open(INFO_FILE))
    try:
        model = XGBClassifier()
        model.load_model(MODEL_FILE)
    except Exception:
        model = _rebuild(info)
    return {**info, "model": model}


def score(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    missing = [c for c in bundle["features"] if c not in df.columns]
    if missing:
        raise SystemExit(f"Input is missing {len(missing)} feature columns, e.g. {missing[:5]}")
    X = df[bundle["features"]].astype(float)
    proba = bundle["model"].predict_proba(X)[:, 1]
    out = df.copy()
    out["bankruptcy_probability"] = proba.round(4)
    out["prediction"] = ["BANKRUPT RISK" if p >= bundle["threshold"] else "safe" for p in proba]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="CSV with columns Attr1..Attr64")
    ap.add_argument("--out", help="where to save scored rows")
    ap.add_argument("-n", type=int, default=10, help="rows to sample in demo mode")
    args = ap.parse_args()

    bundle = load_bundle()
    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        test = pd.read_csv(RESULTS_DIR / "test_set_sample.csv")
        # Mix of real bankrupt and healthy firms so the demo shows both outcomes
        k = max(1, args.n // 2)
        df = pd.concat([test[test.actual_bankrupt == 1].sample(k, random_state=SEED),
                        test[test.actual_bankrupt == 0].sample(args.n - k, random_state=SEED)])
        df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    scored = score(df, bundle)
    print(f"Model: {bundle['name']} | decision threshold {bundle['threshold']:.3f}\n")
    cols = ["bankruptcy_probability", "prediction"] + (["actual_bankrupt"] if "actual_bankrupt" in scored else [])
    view = scored[cols].copy()
    if "actual_bankrupt" in view:
        view["actual_bankrupt"] = view["actual_bankrupt"].map({1: "bankrupt", 0: "survived"})
    print(view.to_string())
    if args.out:
        scored.to_csv(args.out, index=False)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
