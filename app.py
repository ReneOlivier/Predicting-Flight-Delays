
import streamlit as st
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from catboost import CatBoostClassifier

BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_artefacts():
    model = CatBoostClassifier()
    model.load_model(str(BASE_DIR / "models" / "catboost_model.cbm"))
    ct          = joblib.load(BASE_DIR / "models" / "column_transformer.pkl")
    sc          = joblib.load(BASE_DIR / "models" / "scaler.pkl")
    pt_features = joblib.load(BASE_DIR / "models" / "passthrough_features.pkl")
    return model, ct, sc, pt_features

# ── UI ──────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Flight Delay Predictor", page_icon="✈️", layout="centered")
st.title("✈️ Flight Delay Predictor")
st.markdown(
    "Predict the probability that a US domestic flight will arrive **15+ minutes late**, "
    "based on BTS on-time data (January 2020). Model: CatBoost (AUC 0.96)."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    carrier = st.selectbox("Airline carrier", sorted([
        "AA", "AS", "B6", "DL", "F9", "G4", "HA", "MQ",
        "NK", "OH", "OO", "QX", "UA", "WN", "YV", "YX"
    ]))
    origin = st.text_input("Origin airport (IATA)", value="JFK").strip().upper()
    dep_time_blk = st.selectbox("Departure time block", [
        "0001-0559", "0600-0659", "0700-0759", "0800-0859",
        "0900-0959", "1000-1059", "1100-1159", "1200-1259",
        "1300-1359", "1400-1459", "1500-1559", "1600-1659",
        "1700-1759", "1800-1859", "1900-1959", "2000-2059",
        "2100-2159", "2200-2259", "2300-2359"
    ])

with col2:
    dest = st.text_input("Destination airport (IATA)", value="LAX").strip().upper()
    dep_delay = st.number_input("Departure delay (mins, 0 if none)", min_value=0, value=0, step=1)
    distance  = st.number_input("Flight distance (miles)", min_value=50, value=500, step=10)

st.divider()

if st.button("Predict delay probability", type="primary", use_container_width=True):

    # Build a raw DataFrame that matches the structure BEFORE ct.fit_transform
    # Categorical columns first (as ct expects), then passthrough numerics
    cat_cols = ['OP_UNIQUE_CARRIER', 'OP_CARRIER', 'ORIGIN', 'DEST', 'DEP_TIME_BLK']

    # Map user inputs — OP_UNIQUE_CARRIER and OP_CARRIER are the same in this dataset
    cat_values = [carrier, carrier, origin, dest, dep_time_blk]

    # Build the numeric part — must match passthrough_features order exactly
    numeric_input = {
        'DEP_TIME':   int(dep_time_blk[:4]),   # use start of block as proxy
        'DEP_DEL15':  1 if dep_delay >= 15 else 0,
        'DEP_DELAY':  dep_delay,
        'DISTANCE':   distance,
    }
    # Construct the row — categoricals first (matching ct column order), then numerics
    row = dict(zip(cat_cols, cat_values))
    row.update({f: numeric_input.get(f, 0) for f in passthrough_features})

    input_df = pd.DataFrame([row])

    try:
        X_transformed = ct.transform(input_df)
        X_scaled      = sc.transform(X_transformed)
        prob          = model.predict_proba(X_scaled)[0][1]

        if prob >= 0.5:
            st.error(f"⚠️ **Likely delayed** — {prob:.1%} probability of 15+ min arrival delay")
        else:
            st.success(f"✅ **Likely on time** — {prob:.1%} probability of 15+ min arrival delay")

        st.progress(float(prob), text=f"{prob:.1%}")

    except Exception as e:
        st.warning(
            "Could not process that input — the airport code may not have been seen during training. "
            "Try a major hub like JFK, LAX, ORD, DFW, ATL."
        )
        st.caption(f"Error: {e}")

st.divider()
st.caption("Data: BTS On-Time Performance, January 2020 · Model trained on ~450k flights")