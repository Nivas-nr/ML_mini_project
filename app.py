"""Interactive demo.  Run:  streamlit run app.py"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import FEATURE_DESCRIPTIONS, FIGURES_DIR, RESULTS_DIR  # noqa: E402
from predict import load_bundle, score  # noqa: E402

st.set_page_config(page_title="Bankruptcy Predictor", layout="wide")
bundle = load_bundle()
test = pd.read_csv(RESULTS_DIR / "test_set_sample.csv")

st.title("Corporate Bankruptcy Prediction")
st.caption(f"{bundle['name']} trained on the UCI Polish companies data (3rd-year file, outcome 3 years ahead). "
           f"Decision threshold {bundle['threshold']:.2f}, tuned by cross-validation.")

tab_score, tab_results = st.tabs(["Score a firm", "Results"])

with tab_score:
    left, right = st.columns([1, 2])
    with left:
        source = st.radio("Pick a firm from", ["Held-out test set", "Upload CSV"])
        if source == "Held-out test set":
            group = st.selectbox("Show", ["Firms that went bankrupt", "Firms that survived", "Any firm"])
            pool = {"Firms that went bankrupt": test[test.actual_bankrupt == 1],
                    "Firms that survived": test[test.actual_bankrupt == 0]}.get(group, test)
            idx = st.selectbox("Firm (test-set row)", pool.index.tolist())
            firm = test.loc[[idx]]
        else:
            up = st.file_uploader("CSV with columns Attr1..Attr64", type="csv")
            if up is None:
                st.stop()
            firm = pd.read_csv(up)

    with right:
        result = score(firm, bundle)
        for _, row in result.iterrows():
            p = row["bankruptcy_probability"]
            risky = row["prediction"] != "safe"
            st.metric("Bankruptcy probability", f"{p:.1%}",
                      "Above threshold: flag for review" if risky else "Below threshold",
                      delta_color="inverse" if risky else "off")
            st.progress(min(float(p), 1.0))
            if "actual_bankrupt" in row:
                st.write("Actual outcome: **" + ("went bankrupt" if row["actual_bankrupt"] == 1 else "survived") + "**")

        st.subheader("This firm's most influential ratios")
        top = bundle["top_features"]
        median = test[top].median()
        st.dataframe(pd.DataFrame({
            "ratio": [f"{f}: {FEATURE_DESCRIPTIONS[f]}" for f in top],
            "this firm": firm[top].iloc[0].round(3).values,
            "test-set median": median.round(3).values,
        }), hide_index=True, use_container_width=True)

with tab_results:
    cmp = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
    st.dataframe(cmp[["model", "cv_pr_auc_mean", "test_pr_auc", "test_roc_auc", "test_recall",
                      "test_precision", "test_f1", "test_accuracy"]].round(3), hide_index=True,
                 use_container_width=True)
    for f in ["accuracy_trap", "model_comparison", "pr_curves", "confusion_matrix", "feature_importance", "horizon"]:
        path = FIGURES_DIR / f"{f}.png"
        if path.exists():
            st.image(str(path))
