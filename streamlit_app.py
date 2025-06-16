# streamlit_app.py (OzWinner)
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from scipy.stats import entropy

# Constants
NUMBERS_RANGE = list(range(1, 48))
NUM_MAIN = 7
NUM_SUPP = 2

st.title("🎯 OzWinner Lotto Predictor")
st.markdown("Upload historical data and predict the next winning combinations.")

# Upload CSV
csv_file = st.file_uploader("Upload Historical Data (.csv)", type=["csv"])
model_file = st.file_uploader("Upload Trained ML Model (.pkl)", type=["pkl"])

if csv_file:
    df = pd.read_csv(csv_file)
    st.success("CSV uploaded.")
    all_numbers = df.iloc[:, :NUM_MAIN].values.flatten()
    historical_freq = pd.Series(all_numbers).value_counts().sort_index()
else:
    st.stop()

# Load model if available
model = None
if model_file:
    model = joblib.load(model_file)
    st.success("ML model loaded.")

def generate_predictions():
    scores = historical_freq.copy()
    scores = scores.reindex(NUMBERS_RANGE, fill_value=0)
    scores += pd.Series(np.random.randn(len(NUMBERS_RANGE)), index=NUMBERS_RANGE)
    probs = scores / scores.sum()
    predictions = []
    for _ in range(10):
        mains = np.random.choice(NUMBERS_RANGE, size=NUM_MAIN, replace=False, p=probs)
        remaining = list(set(NUMBERS_RANGE) - set(mains))
        supps = np.random.choice(remaining, size=NUM_SUPP, replace=False)
        predictions.append(sorted(mains.tolist()) + sorted(supps.tolist()))
    return predictions

predictions = generate_predictions()

# Show results
pred_df = pd.DataFrame(predictions, columns=[f"N{i+1}" for i in range(NUM_MAIN + NUM_SUPP)])
if model:
    def extract_features(row):
        nums = row[:NUM_MAIN]
        freq_vals = [historical_freq.get(n, 0) for n in nums]
        return [
            np.mean(nums),
            np.std(nums),
            entropy([f / sum(freq_vals) if sum(freq_vals) > 0 else 1 for f in freq_vals])
        ]
    X = np.array([extract_features(r) for r in predictions])
    pred_df["ML Score"] = model.predict(X)

st.dataframe(pred_df)
csv = pred_df.to_csv(index=False).encode("utf-8")
st.download_button("Download Predictions", csv, "ozwinner_predictions.csv")
