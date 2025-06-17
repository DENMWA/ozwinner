
# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.stats import entropy

st.set_page_config(page_title="OzWinner Predictor", layout="centered")
st.title("🎯 OzWinner Lotto Predictor")
st.markdown("Upload historical data and predict the next winning combinations.")

NUMBERS_RANGE = list(range(1, 48))
NUM_MAIN = 7
NUM_PREDICTIONS = 10

uploaded_file = st.file_uploader("Upload Historical Data (.csv)", type=["csv"])
model_file = st.file_uploader("Upload Trained ML Model (.pkl)", type=["pkl"])

df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("CSV uploaded.")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

model = None
if model_file is not None:
    try:
        model = joblib.load(model_file)
        st.success("ML Model loaded.")
    except Exception as e:
        st.error(f"Failed to load model: {e}")

def generate_predictions():
    historical_freq = pd.Series(df.values.flatten())
    historical_freq = historical_freq[historical_freq.isin(NUMBERS_RANGE)]
    historical_freq = historical_freq.value_counts().reindex(NUMBERS_RANGE, fill_value=0).sort_index()
    scores = historical_freq.copy()
    scores += np.random.randn(len(NUMBERS_RANGE)) * 0.5
    probs = scores / scores.sum()
    predictions = []
    for _ in range(NUM_PREDICTIONS):
        mains = np.random.choice(NUMBERS_RANGE, size=NUM_MAIN, replace=False, p=probs)
        predictions.append(sorted(mains))
    return predictions

if df is not None:
    try:
        predictions = generate_predictions()
        st.subheader("🔢 Predicted Entries")
        pred_df = pd.DataFrame(predictions, columns=[f"N{i+1}" for i in range(NUM_MAIN)])
        st.dataframe(pred_df)

        if model is not None:
            def build_features(draw):
                freq = pd.Series(draw).value_counts().reindex(NUMBERS_RANGE, fill_value=0).sort_index()
                return {
                    'mean': np.mean(draw),
                    'std': np.std(draw),
                    'entropy': entropy(freq + 1)  # Laplace smoothing
                }
            features_df = pd.DataFrame([build_features(p) for p in predictions])
            scores = model.predict(features_df)
            pred_df["ML Score"] = np.round(scores, 3)
            st.subheader("🤖 ML Scores")
            st.dataframe(pred_df)

        csv = pred_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Predictions", csv, "ozwinner_predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Prediction Error: {e}")
