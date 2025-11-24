import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import altair as alt

# ============================
# LOAD MODEL & DATA
# ============================

@st.cache_resource
def load_model():
    if os.path.exists("model_ipm_gradientboosting.joblib"):
        loaded = joblib.load("model_ipm_gradientboosting.joblib")
        return loaded["model"], loaded["features"]
    return None, None

@st.cache_data
def load_data():
    if os.path.exists("data_ipm.csv"):
        return pd.read_csv("data_ipm.csv")
    return None

model, feature_names = load_model()
df_hist = load_data()

st.set_page_config(
    page_title="IPM Indonesia – Prediksi & Forecasting",
    layout="wide"
)

# ============================
# FUNGSI UTAMA: DRIFT METHOD
# ============================
def calculate_drift_forecast(df, horizon, feature_cols=["UHH", "HLS", "RLS", "Pengeluaran"]):
    """
    Melakukan forecasting menggunakan metode Drift.
    """
    df = df.sort_values("Tahun")
    
    if len(df) < 2:
        return None, "Data historis harus minimal 2 tahun untuk menghitung drift."

    last_year = int(df["Tahun"].iloc[-1])
    T = len(df)
    
    future_data = []

    for h in range(1, horizon + 1):
        row = {"Tahun": last_year + h}
        
        for col in feature_cols:
            y_T = df[col].iloc[-1]
            y_1 = df[col].iloc[0]
            
            # Rumus Drift: Slope rata-rata dikalikan horizon h
            slope = (y_T - y_1) / (T - 1)
            y_future = y_T + (h * slope)
            
            # Pastikan tidak negatif (terutama untuk HLS/RLS jika tren turun drastis)
            row[col] = max(0, y_future)
            
        future_data.append(row)
    
    return pd.DataFrame(future_data), None

# ============================
# HEADER
# ============================

st.title("📈 Prediksi & Forecasting IPM Indonesia")
st.write(f"Model siap: {'✅' if model else '❌'} | Fitur Forecasting hingga 2030 dengan Metode Drift.")

# ============================
# SIDEBAR
# ============================
with st.sidebar:
    st.title("Menu Utama")
    st.info("Gunakan tab di sebelah kanan untuk forecasting massal.")

# ============================
# TAB UTAMA
# ============================

tab1, tab2, tab3 = st.tabs([
    "📊 Visualisasi Historis",
    "🔮 Forecast IPM (Target 2030)",
    "📤 Upload & Prediksi Massal"
])

# ======================================
# TAB 1 – VISUALISASI HISTORIS
# ======================================
with tab1:
    st.subheader("📊 Ringkasan Data Historis")
    if df_hist is None:
        st.warning("File data_ipm.csv tidak ditemukan. Silakan upload data di Tab 2 untuk forecasting.")
    else:
        st.dataframe(df_hist.head())
        chart = alt.Chart(df_hist).mark_line(point=True).encode(
            x='Tahun:O', 
            y=alt.Y('IPM', scale=alt.Scale(zero=False)),
            color='Cakupan',
            tooltip=['Tahun', 'Cakupan', 'IPM']
        ).interactive()
        st.altair_chart(chart, use_container_width=True)

# ======================================
# TAB 2 – FORECAST IPM (DRIFT METHOD)
# ======================================
with tab2:
    st.subheader("🔮 Forecast IPM hingga Tahun Target (Metode Drift)")
    
    # 1. SUMBER DATA
