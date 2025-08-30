import streamlit as st
import pandas as pd
import numpy as np
import dill
import json
import os

st.set_page_config(page_title="Responsible Attrition EWS", layout="wide")

# --- Load artifacts ---
@st.cache_resource
def load_all():
    with open("artifacts/model_pipeline.dill", "rb") as f:
        pipe = dill.load(f)
    data = pd.read_csv("artifacts/dataset_with_synth.csv")
    risk_holdout = pd.read_csv("artifacts/risk_scores_holdout.csv") if os.path.exists("artifacts/risk_scores_holdout.csv") else None
    meta = json.load(open("artifacts/meta.json"))
    return pipe, data, risk_holdout, meta

pipe, data, risk_holdout, meta = load_all()
num_cols, cat_cols = meta["num_cols"], meta["cat_cols"]
RISK_THRESHOLDS = meta["risk_thresholds"]

# --- Prediction and bucket functions ---
def predict(df):
    return pipe.predict_proba(df)[:, 1]

def bucket(p):
    return "High" if p >= RISK_THRESHOLDS["High"] else ("Medium" if p >= RISK_THRESHOLDS["Medium"] else "Low")

def generate_nudges(row: pd.Series) -> list:
    n = []
    if "OverTime" in row and str(row["OverTime"]).lower() == "yes":
        n.append("Discuss workload & overtime; consider resource rebalancing or comp-off.")
    if "JobSatisfaction" in row and row["JobSatisfaction"] <= 2:
        n.append("Schedule a career growth conversation and clarify role expectations.")
    if "RecognitionCount6M" in row and row["RecognitionCount6M"] <= 1:
        n.append("Provide timely recognition for recent contributions.")
    if "LMS_TrainSessions6M" in row and row["LMS_TrainSessions6M"] <= 1:
        n.append("Recommend a skill course aligned to career goals (opt-in).")
    if "PromotionWaitVsPeers" in row and row["PromotionWaitVsPeers"] >= 1:
        n.append("Discuss promotion path with clear timelines.")
    if "CompHikePctVsPeers" in row and row["CompHikePctVsPeers"] <= -2:
        n.append("Review compensation fairness vs peers.")
    if "LeaveUtilization6M" in row and row["LeaveUtilization6M"] < 0.3:
        n.append("Encourage planned time-off to avoid burnout.")
    if "TeamInteractionScore" in row and row["TeamInteractionScore"] < -0.5:
        n.append("Facilitate team check-ins/1:1s.")
    if "ExitPortalLogin" in row and row["ExitPortalLogin"] == 1:
        n.append("[Sensitive] Offer confidential career discussion (only if consented).")
    return n[:5] if n else ["Check-in: ask about workload, growth, recognition, and support needed."]

# --- Streamlit layout ---
st.title("Responsible AI - Early Warning System for Attrition/Disengagement")

tab1, tab2 = st.tabs(["Overview", "Profile"])

with tab1:
    st.subheader("Risk overview")
    scores = predict(data)
    buckets = np.array([bucket(p) for p in scores])
    view = data.copy()
    view["RiskScore"] = scores
    view["RiskBucket"] = buckets

    dept = st.selectbox("Filter by Department", ["All"] + sorted(view["Department"].dropna().unique().tolist()) if "Department" in view.columns else ["All"])
    bucket_f = st.multiselect("Risk Bucket", ["High", "Medium", "Low"], default=[])
    if dept != "All":
        view = view[view["Department"] == dept]
    view = view[view["RiskBucket"].isin(bucket_f)]

    st.dataframe(view.sort_values("RiskScore", ascending=False).head(200), use_container_width=True)
    st.markdown("**Fairness note:** Predictions are calibrated probabilities. Combine with human judgment and employee consent.")

with tab2:
    st.subheader("Employee profile")
    idx = st.number_input("Row index", min_value=0, max_value=len(data)-1, value=0, step=1)
    row = data.iloc[int(idx)]
    rs = float(predict(pd.DataFrame([row]))[0])
    rb = bucket(rs)
    st.metric("Risk Score", f"{rs:.3f}")
    st.metric("Bucket", rb)

    st.markdown("### Recommended actions")
    for i, n in enumerate(generate_nudges(row), 1):
        st.write(f"{i}. {n}")

    st.markdown("### Add offline observation")
    obs = st.text_area("Note (what you discussed / agreed next steps)")
    if st.button("Save observation"):
        obs_df = pd.DataFrame([{"index": int(idx), "risk_score": rs, "bucket": rb, "note": obs}])
        os.makedirs("artifacts", exist_ok=True)
        path = "artifacts/observations.csv"
        if os.path.exists(path):
            old = pd.read_csv(path)
            pd.concat([old, obs_df], ignore_index=True).to_csv(path, index=False)
        else:
            obs_df.to_csv(path, index=False)
        st.success("Saved.")
