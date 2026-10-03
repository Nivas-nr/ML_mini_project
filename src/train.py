"""Run every experiment, save results tables, figures and the final model.

Usage:  python src/train.py            (full run, a few minutes)
        python src/train.py --quick    (skips SVM and the 5-year horizon study)
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve

import plots
from config import (FEATURE_DESCRIPTIONS, FIGURES_DIR, HORIZON_YEARS, MAIN_YEAR, MODELS_DIR,
                    RESULTS_DIR, SEED)
from data import describe, load_year, split
from evaluate import best_f1_threshold, cross_validate, scores
from models import build_imbalance_variants, build_models
from plots import AQUA, BLUE, GRAY, INK_2, ORANGE, plt

FINAL_MODEL = "XGBoost"  # chosen from the CV results below; see README


def pos_weight(y):
    return float((y == 0).sum() / (y == 1).sum())


def compare_models(X_tr, y_tr, X_te, y_te, skip_svm=False):
    rows, curves, fitted = [], {}, {}
    for name, model in build_models(pos_weight(y_tr)).items():
        if skip_svm and name.startswith("SVM"):
            continue
        t0 = time.time()
        oof, folds = cross_validate(model, X_tr, y_tr)
        thr = 0.5 if name.startswith("Baseline") else best_f1_threshold(y_tr, oof)
        final = clone(model).fit(X_tr, y_tr)
        proba = final.predict_proba(X_te)[:, 1]
        secs = time.time() - t0

        cv_pr = [f["pr_auc"] for f in folds]
        cv_roc = [f["roc_auc"] for f in folds]
        at_05 = scores(y_te, proba, 0.5)
        tuned = scores(y_te, proba, thr)
        rows.append({
            "model": name,
            "cv_pr_auc_mean": np.mean(cv_pr), "cv_pr_auc_std": np.std(cv_pr),
            "cv_roc_auc_mean": np.mean(cv_roc), "cv_roc_auc_std": np.std(cv_roc),
            "threshold": thr,
            "test_accuracy@0.5": at_05["accuracy"], "test_recall@0.5": at_05["recall"],
            **{f"test_{k}": v for k, v in tuned.items()},
            "seconds": secs,
        })
        curves[name] = proba
        fitted[name] = final
        print(f"  {name:<26} CV PR-AUC {np.mean(cv_pr):.3f}±{np.std(cv_pr):.3f} | "
              f"test PR-AUC {tuned['pr_auc']:.3f} ROC-AUC {tuned['roc_auc']:.3f} "
              f"recall {tuned['recall']:.2f} precision {tuned['precision']:.2f} ({secs:.0f}s)")
    return pd.DataFrame(rows), curves, fitted


def imbalance_study(X_tr, y_tr, X_te, y_te):
    rows = []
    for name, model in build_imbalance_variants(pos_weight(y_tr)).items():
        oof, folds = cross_validate(model, X_tr, y_tr)
        proba = clone(model).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
        s05 = scores(y_te, proba, 0.5)
        thr = best_f1_threshold(y_tr, oof)
        st = scores(y_te, proba, thr)
        rows.append({"strategy": name, "cv_pr_auc": np.mean([f["pr_auc"] for f in folds]),
                     "test_pr_auc": st["pr_auc"], "recall@0.5": s05["recall"], "f1@0.5": s05["f1"],
                     "threshold": thr, "recall@tuned": st["recall"], "f1@tuned": st["f1"]})
        print(f"  {name:<20} recall@0.5 {s05['recall']:.2f} | recall@tuned {st['recall']:.2f} f1@tuned {st['f1']:.2f}")
    return pd.DataFrame(rows)


def horizon_study():
    rows = []
    for year in range(1, 6):
        X, y = load_year(year)
        X_tr, X_te, y_tr, y_te = split(X, y)
        model = build_models(pos_weight(y_tr))[FINAL_MODEL]
        oof, _ = cross_validate(model, X_tr, y_tr)
        thr = best_f1_threshold(y_tr, oof)
        proba = clone(model).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
        s = scores(y_te, proba, thr)
        rows.append({"file": f"{year}year", "years_before_bankruptcy": HORIZON_YEARS[year],
                     "rows": len(X), "bankrupt_pct": 100 * y.mean(),
                     "test_roc_auc": s["roc_auc"], "test_pr_auc": s["pr_auc"],
                     "test_recall": s["recall"], "test_precision": s["precision"], "test_f1": s["f1"]})
        print(f"  {year}year ({HORIZON_YEARS[year]}y ahead): ROC-AUC {s['roc_auc']:.3f} PR-AUC {s['pr_auc']:.3f} recall {s['recall']:.2f}")
    return pd.DataFrame(rows).sort_values("years_before_bankruptcy")


# ---------------------------------------------------------------- figures
def fig_model_comparison(res, prevalence):
    d = res.sort_values("test_pr_auc")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    colors = [GRAY if m.startswith("Baseline") else BLUE for m in d["model"]]
    ax.barh(d["model"], d["test_pr_auc"], color=colors, height=0.6)
    for y, v in enumerate(d["test_pr_auc"]):
        ax.text(v + 0.01, y, f"{v:.2f}", va="center", color=INK_2, fontsize=10)
    ax.axvline(prevalence, color=INK_2, lw=1, ls="--")
    ax.text(prevalence + 0.01, len(d) - 0.45, f"random guess = {prevalence:.2f}", color=INK_2, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("PR-AUC on held-out test set (higher is better)")
    ax.set_title("Model comparison: how well each model ranks bankrupt firms")
    ax.grid(axis="y", visible=False)
    plots.save(fig, FIGURES_DIR / "model_comparison.png")


def fig_accuracy_trap(res):
    pick = ["Baseline (always 'safe')", "Logistic Regression", "Random Forest", FINAL_MODEL]
    d = res.set_index("model").loc[[p for p in pick if p in set(res["model"])]]
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    w = 0.36
    ax.bar(x - w / 2 - 0.01, d["test_accuracy"], w, color=BLUE, label="Accuracy")
    ax.bar(x + w / 2 + 0.01, d["test_recall"], w, color=ORANGE, label="Recall on bankrupt firms")
    for i, (a, r) in enumerate(zip(d["test_accuracy"], d["test_recall"])):
        ax.text(i - w / 2, a + 0.02, f"{a:.0%}", ha="center", fontsize=9, color=INK_2)
        ax.text(i + w / 2, r + 0.02, f"{r:.0%}", ha="center", fontsize=9, color=INK_2)
    ax.set_xticks(x, [m.replace(" (always 'safe')", "\n(always 'safe')") for m in d.index])
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("The accuracy trap: 95% accuracy can mean zero bankruptcies caught")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.14))
    ax.grid(axis="x", visible=False)
    plots.save(fig, FIGURES_DIR / "accuracy_trap.png")


def fig_pr_curves(curves, y_te, res):
    top = res[~res["model"].str.startswith("Baseline")].nlargest(3, "test_pr_auc")["model"].tolist()
    fig, ax = plt.subplots(figsize=(6.4, 5))
    for name, c in zip(top, [BLUE, ORANGE, AQUA]):
        p, r, _ = precision_recall_curve(y_te, curves[name])
        ax.plot(r, p, color=c, lw=2, label=f"{name} (PR-AUC {res.set_index('model').loc[name, 'test_pr_auc']:.2f})")
    ax.axhline(y_te.mean(), color=GRAY, lw=1, ls="--", label=f"Random guess ({y_te.mean():.2f})")
    ax.set_xlabel("Recall (share of bankrupt firms caught)")
    ax.set_ylabel("Precision (share of alerts that are correct)")
    ax.set_xlim(0, 1), ax.set_ylim(0, 1.02)
    ax.set_title("Precision-recall curves, top 3 models")
    ax.legend(loc="lower left", fontsize=9)
    plots.save(fig, FIGURES_DIR / "pr_curves.png")


def fig_confusion(row):
    m = np.array([[row["test_tn"], row["test_fp"]], [row["test_fn"], row["test_tp"]]]).astype(int)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(m / m.sum(axis=1, keepdims=True), cmap=matplotlib_blues(), vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            frac = m[i, j] / m[i].sum()
            ax.text(j, i, f"{m[i, j]}\n({frac:.0%} of row)", ha="center", va="center",
                    color="white" if frac > 0.55 else "#0b0b0b", fontsize=11)
    ax.set_xticks([0, 1], ["Predicted safe", "Predicted bankrupt"])
    ax.set_yticks([0, 1], ["Actually safe", "Actually bankrupt"])
    ax.grid(False)
    ax.set_title(f"{FINAL_MODEL} on test set (threshold {row['threshold']:.2f})", fontsize=11)
    plots.save(fig, FIGURES_DIR / "confusion_matrix.png")


def matplotlib_blues():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seq_blue", ["#eef5fd", "#86b6ef", "#2a78d6", "#104281"])


def fig_importance(model, X_te, y_te):
    pi = permutation_importance(model, X_te, y_te, scoring="average_precision", n_repeats=10,
                                random_state=SEED, n_jobs=-1)
    imp = pd.DataFrame({"feature": X_te.columns, "importance": pi.importances_mean, "std": pi.importances_std})
    imp["description"] = imp["feature"].map(FEATURE_DESCRIPTIONS)
    imp = imp.sort_values("importance", ascending=False)
    imp.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

    top = imp.head(10).iloc[::-1]
    labels = [f"{f}: {d if len(d) <= 48 else d[:46] + '…'}" for f, d in zip(top["feature"], top["description"])]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(labels, top["importance"], color=BLUE, height=0.6)
    ax.set_xlabel("Drop in test PR-AUC when the feature is shuffled")
    ax.set_title(f"Top 10 financial ratios driving the {FINAL_MODEL} model")
    ax.grid(axis="y", visible=False)
    plots.save(fig, FIGURES_DIR / "feature_importance.png")
    return imp


def fig_horizon(h):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = h["years_before_bankruptcy"]
    ax.plot(x, h["test_roc_auc"], color=BLUE, lw=2, marker="o", ms=8, label="ROC-AUC")
    ax.plot(x, h["test_pr_auc"], color=ORANGE, lw=2, marker="o", ms=8, label="PR-AUC")
    for xi, a, b in zip(x, h["test_roc_auc"], h["test_pr_auc"]):
        ax.text(xi, a + 0.03, f"{a:.2f}", ha="center", fontsize=9, color=INK_2)
        ax.text(xi, b - 0.07, f"{b:.2f}", ha="center", fontsize=9, color=INK_2)
    ax.set_xticks(x)
    ax.set_xlabel("Years before the bankruptcy outcome")
    ax.set_ylim(0, 1.08)
    ax.invert_xaxis()
    ax.set_title(f"{FINAL_MODEL} stays strong at every forecast horizon")
    ax.legend(loc="lower left")
    plots.save(fig, FIGURES_DIR / "horizon.png")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip SVM and the horizon study")
    args = ap.parse_args()

    for d in (RESULTS_DIR, FIGURES_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    plots.setup()

    X, y = load_year(MAIN_YEAR)
    X_tr, X_te, y_tr, y_te = split(X, y)
    info = {"full": describe(X, y), "train": describe(X_tr, y_tr), "test": describe(X_te, y_te)}
    print(f"Data ({MAIN_YEAR}year.arff, duplicates removed):", json.dumps(info, indent=1))

    print("\n[1/3] Comparing models (5-fold stratified CV on train, then held-out test)")
    res, curves, fitted = compare_models(X_tr, y_tr, X_te, y_te, skip_svm=args.quick)
    res.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    print("\n[2/3] Class-imbalance strategies (XGBoost)")
    imb = imbalance_study(X_tr, y_tr, X_te, y_te)
    imb.to_csv(RESULTS_DIR / "imbalance_study.csv", index=False)

    final_row = res.set_index("model").loc[FINAL_MODEL]
    fig_model_comparison(res, y_te.mean())
    fig_accuracy_trap(res)
    fig_pr_curves(curves, y_te, res)
    fig_confusion(final_row)
    imp = fig_importance(fitted[FINAL_MODEL], X_te, y_te)

    if not args.quick:
        print("\n[3/3] Forecast horizon: same model on each of the 5 yearly files")
        h = horizon_study()
        h.to_csv(RESULTS_DIR / "horizon_study.csv", index=False)
        fig_horizon(h)

    # Final model for the demo: trained on the training split, threshold from CV.
    # Saved in XGBoost's own JSON format (portable across operating systems and Python versions)
    # plus a small metadata file, instead of a pickle.
    fitted[FINAL_MODEL].save_model(MODELS_DIR / "xgboost_model.json")
    json.dump({"name": FINAL_MODEL, "threshold": float(final_row["threshold"]),
               "features": list(X.columns), "top_features": imp["feature"].head(10).tolist()},
              open(MODELS_DIR / "model_info.json", "w"), indent=2)
    X_te.assign(actual_bankrupt=y_te.values).to_csv(RESULTS_DIR / "test_set_sample.csv", index=False)
    json.dump(info, open(RESULTS_DIR / "data_summary.json", "w"), indent=2)
    print(f"\nDone. Results in {RESULTS_DIR}, model in {MODELS_DIR}")


if __name__ == "__main__":
    main()
