import pandas as pd
import numpy as np
import time
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import streamlit as st

# Simulate incoming data
def generate_data(n=200):
    normal = np.random.normal(50, 5, n)
    anomalies = np.random.choice([100, 5, 120, -10], size=int(n*0.05))
    data = normal.copy()
    anomaly_indices = np.random.choice(range(n), size=len(anomalies), replace=False)
    data[anomaly_indices] = anomalies
    return pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=n, freq="min"),
                         "value": data})

df = generate_data()
model = IsolationForest(contamination=0.05, random_state=42)
df["anomaly"] = model.fit_predict(df[["value"]])
df["alert"] = df["anomaly"].apply(lambda x: "⚠️ ALERT" if x == -1 else "")

st.title("🚀 Real-time Data Analysis Demo")
st.line_chart(df.set_index("timestamp")["value"])
st.write("### Alerts Detected:")
st.dataframe(df[df["alert"] != ""].head(10))
