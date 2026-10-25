
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Bank Marketing Predictor", page_icon="🏦", layout="centered")
st.title("🏦 Bank Term Deposit Subscription Predictor")
st.caption("Educational demonstration using historic Portuguese bank campaign data. Not a decision system for customer eligibility.")

@st.cache_resource
def load_assets():
    folder = Path(__file__).parent
    trained = joblib.load(folder / "model.joblib")
    information = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    return trained, information

try:
    model, meta = load_assets()
except FileNotFoundError:
    st.error("Missing model.joblib or metadata.json. Run the training notebook and download both files.")
    st.stop()

st.write("Selected model:", meta["algorithm"])
st.warning("Estimates are based on historical data and may be unreliable for present-day customers. Do not use as an automated eligibility decision.")
mode = st.radio("Prediction mode", ["One customer", "Upload CSV"], horizontal=True)

if mode == "One customer":
    values = {}
    with st.form("prediction"):
        left, right = st.columns(2)
        for i, feature in enumerate(meta["features"]):
            with (left if i % 2 == 0 else right):
                if feature in meta["categorical"]:
                    values[feature] = st.selectbox(feature, meta["categories"][feature])
                else:
                    values[feature] = st.number_input(feature, value=float(meta["defaults"][feature]), format="%.4f")
        submitted = st.form_submit_button("Predict subscription")
    if submitted:
        one = pd.DataFrame([values], columns=meta["features"])
        predicted = int(model.predict(one)[0])
        if predicted:
            st.success("Predicted: likely to subscribe")
        else:
            st.info("Predicted: unlikely to subscribe")
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(one)[0][1])
            st.metric("Estimated probability of subscription", f"{probability:.1%}")
            st.caption("Model-estimated probability, not a guaranteed or necessarily calibrated chance.")
else:
    uploaded = st.file_uploader("CSV with the exact training input columns (no target required)", type="csv")
    if uploaded is not None:
        incoming = pd.read_csv(uploaded)
        missing = [c for c in meta["features"] if c not in incoming.columns]
        if missing:
            st.error(f"CSV is missing required columns: {missing}")
        else:
            output = incoming.copy()
            output["prediction"] = model.predict(incoming[meta["features"]])
            st.dataframe(output.head(25), hide_index=True)
            st.download_button("Download predictions.csv", output.to_csv(index=False),
                               file_name="predictions.csv", mime="text/csv")

st.caption("Dataset: Moro, Cortez and Rita / UCI Bank Marketing. Call duration is excluded to prevent leakage.")
