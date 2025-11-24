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
    # Pastikan file model ada di direktori yang sama
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
def calculate_drift_forecast(df, horizon=5, feature_cols=["UHH", "HLS", "RLS", "Pengeluaran"]):
    """
    Melakukan forecasting menggunakan metode Drift.
    Rumus: y(T+h) = y(T) + h * ((y(T) - y(1)) / (T - 1))
    """
    # Pastikan data terurut berdasarkan tahun
    df = df.sort_values("Tahun")
    
    if len(df) < 2:
        return None, "Data historis harus minimal 2 tahun untuk menghitung drift."

    last_year = df["Tahun"].iloc[-1]
    T = len(df)
    
    # Dictionary untuk menyimpan hasil forecast per tahun
    future_data = []

    for h in range(1, horizon + 1):
        row = {"Tahun": last_year + h}
        
        for col in feature_cols:
            y_T = df[col].iloc[-1]  # Nilai terakhir
            y_1 = df[col].iloc[0]   # Nilai pertama
            
            # Hitung slope (drift)
            slope = (y_T - y_1) / (T - 1)
            
            # Hitung nilai masa depan
            y_future = y_T + (h * slope)
            row[col] = max(0, y_future) # Cegah nilai negatif
            
        future_data.append(row)
    
    return pd.DataFrame(future_data), None

# ============================
# HEADER
# ============================

st.title("📈 Prediksi & Forecasting IPM Indonesia")
st.write(
    "Aplikasi ini menggunakan model **Gradient Boosting** untuk memprediksi IPM. "
    "Fitur forecasting menggunakan **Metode Drift** untuk memproyeksikan komponen masa depan."
)

# ============================
# SIDEBAR – INPUT MANUAL
# ============================

with st.sidebar:
    st.title("IPM Explorer")
    st.markdown("---")
    st.subheader("🔧 Simulator Prediksi Single")
    
    with st.form("simulation_form"):
        st.write("**Masukkan Indikator:**")
        tahun = st.number_input("Tahun", 2000, 2100, 2025, 1)
        col_sb1, col_sb2 = st.columns(2)
        with col_sb1:
            uhh = st.number_input("UHH (Thn)", 40.0, 90.0, 73.0, 0.1)
            hls = st.number_input("HLS (Thn)", 0.0, 25.0, 13.0, 0.1)
        with col_sb2:
            rls = st.number_input("RLS (Thn)", 0.0, 25.0, 9.0, 0.1)
        pengeluaran = st.number_input("Pengeluaran/Kapita", 0.0, value=12_000_000.0, step=100_000.0, format="%.0f")
        submitted = st.form_submit_button("🚀 Hitung Prediksi", use_container_width=True)

    if submitted and model:
        input_data = pd.DataFrame([{"UHH": uhh, "HLS": hls, "RLS": rls, "Pengeluaran": pengeluaran, "Tahun": tahun}])
        if feature_names: input_data = input_data[feature_names]
        try:
            pred = model.predict(input_data)[0]
            st.markdown("### Hasil Prediksi")
            st.metric(label="IPM Diprediksi", value=f"{pred:.2f}")
            color = "green" if pred >= 80 else "blue" if pred >= 70 else "orange" if pred >= 60 else "red"
            status = "Sangat Tinggi" if pred >= 80 else "Tinggi" if pred >= 70 else "Sedang" if pred >= 60 else "Rendah"
            st.markdown(f"Status: :**{color}[{status}]**")
        except Exception as e: st.error(f"Error: {e}")

# ============================
# TAB UTAMA
# ============================

tab1, tab2, tab3 = st.tabs([
    "📊 Visualisasi Historis",
    "🔮 Forecast IPM (Drift Method)",
    "📤 Upload Data & Prediksi Massal"
])

# ======================================
# TAB 1 – VISUALISASI
# ======================================
with tab1:
    st.subheader("📊 Ringkasan IPM Nasional & Per Daerah")
    if df_hist is None:
        st.warning("File data_ipm.csv tidak ditemukan.")
    else:
        # Visualisasi sederhana (sama seperti sebelumnya, disederhanakan untuk brevity)
        df_nat = df_hist.groupby("Tahun")["IPM"].mean().reset_index()
        chart = alt.Chart(df_nat).mark_line(point=True).encode(
            x='Tahun:O', y=alt.Y('IPM', scale=alt.Scale(zero=False)),
            tooltip=['Tahun', alt.Tooltip('IPM', format='.2f')]
        ).interactive()
        st.altair_chart(chart, use_container_width=True)

# ======================================
# TAB 2 – FORECAST IPM (DRIFT METHOD)
# ======================================
with tab2:
    st.subheader("🔮 Forecast IPM Masa Depan (Metode Drift)")
    st.info("Metode Drift: Garis lurus ditarik dari data pertama ke data terakhir untuk memproyeksikan masa depan.")

    source_option = st.radio("Pilih Sumber Data:", ["Gunakan Data Historis (data_ipm.csv)", "Upload File CSV Baru"])

    df_source = None
    
    if source_option == "Gunakan Data Historis (data_ipm.csv)":
        if df_hist is not None:
            daerah_list = sorted(df_hist["Cakupan"].unique().tolist())
            daerah_pilih = st.selectbox("Pilih Daerah", daerah_list)
            df_source = df_hist[df_hist["Cakupan"] == daerah_pilih].copy()
        else:
            st.error("Data historis tidak ditemukan.")
            
    else: # Upload CSV Baru
        uploaded_forecast = st.file_uploader("Upload file CSV (Format: Cakupan, UHH, HLS, RLS, Pengeluaran, Tahun)", key="up_forecast")
        if uploaded_forecast:
            try:
                # Handle koma sebagai desimal
                df_source = pd.read_csv(uploaded_forecast, decimal=",", thousands=".")
                # Jika user upload banyak daerah, minta pilih satu untuk divisualisasikan
                if "Cakupan" in df_source.columns and df_source["Cakupan"].nunique() > 1:
                    daerah_list = sorted(df_source["Cakupan"].unique().tolist())
                    daerah_pilih = st.selectbox("Pilih Daerah dari File", daerah_list)
                    df_source = df_source[df_source["Cakupan"] == daerah_pilih].copy()
            except Exception as e:
                st.error(f"Gagal membaca file: {e}")

    # PROSES FORECASTING
    if df_source is not None and not df_source.empty:
        horizon = st.slider("Jumlah tahun ke depan:", 1, 15, 5)
        
        # Hitung Forecast Komponen (Drift)
        df_future_comp, err = calculate_drift_forecast(df_source, horizon=horizon)
        
        if err:
            st.warning(err)
        else:
            # Hitung Prediksi IPM dengan Model ML
            if model:
                # Pastikan urutan kolom sesuai training
                X_future = df_future_comp[feature_names]
                df_future_comp["IPM_Prediksi"] = model.predict(X_future)
                
                # Tampilkan Hasil
                st.write(f"**Hasil Forecast {horizon} Tahun ke Depan:**")
                st.dataframe(df_future_comp, use_container_width=True)
                
                # Plotting Gabungan
                df_hist_plot = df_source[["Tahun", "IPM"] if "IPM" in df_source.columns else ["Tahun"]].copy()
                if "IPM" in df_source.columns:
                    df_hist_plot["Jenis"] = "Aktual"
                    df_hist_plot.rename(columns={"IPM": "Nilai"}, inplace=True)
                
                df_future_plot = df_future_comp[["Tahun", "IPM_Prediksi"]].copy()
                df_future_plot["Jenis"] = "Forecast (Drift)"
                df_future_plot.rename(columns={"IPM_Prediksi": "Nilai"}, inplace=True)
                
                df_plot = pd.concat([df_hist_plot, df_future_plot], ignore_index=True)
                
                chart_fore = alt.Chart(df_plot).encode(
                    x='Tahun:O',
                    y=alt.Y('Nilai:Q', scale=alt.Scale(zero=False)),
                    color='Jenis:N',
                    tooltip=['Tahun', 'Jenis', alt.Tooltip('Nilai', format='.2f')]
                ).mark_line(point=True).interactive()
                
                st.altair_chart(chart_fore, use_container_width=True)

# ======================================
# TAB 3 – UPLOAD & ISI PREDIKSI (EXISTING)
# ======================================
with tab3:
    st.subheader("📤 Prediksi Massal (Isi Nilai IPM)")
    st.write("Gunakan fitur ini jika Anda punya data UHH, HLS, dll tapi **belum ada nilai IPM-nya**.")
    
    up_massal = st.file_uploader("Upload Data untuk Diprediksi", type=["csv"], key="up_massal")
    
    if up_massal and model:
        try:
            df_massal = pd.read_csv(up_massal, decimal=",", thousands=".")
            missing = [c for c in feature_names if c not in df_massal.columns]
            
            if missing:
                st.error(f"Kolom hilang: {missing}")
            else:
                df_massal["IPM_Prediksi"] = model.predict(df_massal[feature_names])
                st.success("Selesai!")
                st.dataframe(df_massal)
                
                csv = df_massal.to_csv(index=False).encode('utf-8')
                st.download_button("Download Hasil", csv, "hasil_prediksi.csv", "text/csv")
                
        except Exception as e:
            st.error(f"Error: {e}")
