
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="OzWinner Lotto Predictor")

st.title("🎯 OzWinner Lotto Predictor")
st.markdown("Upload historical data and predict the next winning combinations.")

# Upload CSV
csv_file = st.file_uploader("Upload Historical Data (.csv)", type=["csv"])
model_file = st.file_uploader("Upload Trained ML Model (.pkl)", type=["pkl"])

if csv_file:
    df = pd.read_csv(csv_file)
    st.success("CSV uploaded.")
    st.dataframe(df.head())
else:
    st.warning("Please upload a CSV file to continue.")
    st.stop()

# Feature engineering
def build_features(row):
    row = sorted(row)
    features = {
        "mean": np.mean(row),
        "std": np.std(row),
        "min": np.min(row),
        "max": np.max(row),
        "range": np.ptp(row),
    }
    return features

feature_data = pd.DataFrame([build_features(row[1:8]) for row in df.itertuples()], dtype=np.float32)

# Load model
model = None
if model_file:
    try:
        model = joblib.load(model_file)
        st.success("Model loaded.")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()
else:
    st.warning("Upload a trained ML model to continue.")
    st.stop()

# Generate predictions
def generate_predictions():
    predictions = []
    for _ in range(10):
        sample = df.sample(1).values.flatten()[1:8]
        new_set = np.random.choice(range(1, 48), size=7, replace=False)
        features = build_features(new_set)
        pred_df = pd.DataFrame([features])
        score = model.predict(pred_df)[0]
        predictions.append((sorted(new_set), round(score, 2)))
    return sorted(predictions, key=lambda x: x[1], reverse=True)

results = generate_predictions()
st.subheader("🔮 Top Predicted Sets")
for numbers, score in results:
    st.write(f"Numbers: {numbers} | Score: {score}")
