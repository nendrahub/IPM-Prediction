import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import altair as alt  # Wajib import Altair

# ============================
# KONFIGURASI HALAMAN
# ============================
st.set_page_config(
    page_title="IPM Indonesia Dashboard",
    page_icon="🇮🇩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
# CUSTOM CSS
# ============================
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stDownloadButton button {
        background-color: #2c3e50;
        color: white;
        border-radius: 8px;
    }
    h1, h2, h3 { color: #2c3e50; }
    </style>
""", unsafe_allow_html=True)

# ============================
# LOAD MODEL & DATA
# ============================
@st.cache_resource
def load_model():
    model_path = "model_ipm_gradientboosting.joblib"
    if not os.path.exists(model_path):
        return None, None
    try:
        loaded = joblib.load(model_path)
        if isinstance(loaded, dict):
            return loaded.get("model"), loaded.get("features")
        return loaded, None 
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None

@st.cache_data
def load_data():
    if os.path.exists("data_ipm.csv"):
        return pd.read_csv("data_ipm.csv")
    return None

model, feature_names = load_model()
df_hist = load_data()

# ============================
# SIDEBAR – INPUT MANUAL
# ============================
with st.sidebar:
    st.title("IPM Explorer")
    st.markdown("---")
    st.subheader("🔧 Simulator Prediksi")
    
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
# HEADER UTAMA
# ============================
st.title("📈 Analisis & Forecasting IPM Indonesia")
st.markdown("---")

# ============================
# FUNGSI CHART (DIPERBAIKI)
# ============================
def create_labeled_line_chart(data, x_col, y_col, color_col=None):
    # Base Chart
    base = alt.Chart(data).encode(
        # -----------------------------------------------------------------
        # PERBAIKAN DI SINI:
        # format='d' memaksa angka menjadi desimal bulat (tanpa koma).
        # tickMinStep=1 memastikan tidak ada tahun desimal (misal 2022.5)
        # -----------------------------------------------------------------
        x=alt.X(f'{x_col}:Q', axis=alt.Axis(format='d', tickMinStep=1, title=x_col)), 
        y=alt.Y(f'{y_col}:Q', scale=alt.Scale(zero=False)), 
        tooltip=[alt.Tooltip(x_col, format='d'), alt.Tooltip(y_col, format=".2f")]
    )

    if color_col:
        base = base.encode(color=color_col)
    else:
        base = base.encode(color=alt.value("#2980b9")) 

    line = base.mark_line(strokeWidth=3)
    points = base.mark_point(filled=True, size=50)
    text = base.mark_text(align='center', baseline='bottom', dy=-10, fontSize=12).encode(
        text=alt.Text(y_col, format=".2f")
    )

    return (line + points + text).properties(height=350).interactive()

# ============================
# KONTEN TABS
# ============================
if model is None or df_hist is None:
    st.warning("⚠️ Model atau Data tidak ditemukan.")

tab1, tab2, tab3 = st.tabs(["📊 Dashboard Historis", "🔮 Forecasting Wilayah", "📤 Analisis Massal"])

# -------------------------------------------------------
# TAB 1: DASHBOARD HISTORIS
# -------------------------------------------------------
with tab1:
    if df_hist is not None:
        latest_year = df_hist["Tahun"].max()
        df_latest = df_hist[df_hist["Tahun"] == latest_year]
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rata-rata IPM", f"{df_latest['IPM'].mean():.2f}")
        m2.metric("IPM Tertinggi", f"{df_latest['IPM'].max():.2f}")
        m3.metric("IPM Terendah", f"{df_latest['IPM'].min():.2f}")
        m4.metric("Total Wilayah", f"{df_latest['Cakupan'].nunique()}")
        
        st.markdown("---")
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("##### 📈 Tren Rata-rata IPM Nasional")
            df_trend = df_hist.groupby("Tahun")["IPM"].mean().reset_index()
            # Panggil fungsi chart yang sudah diperbaiki
            chart_trend = create_labeled_line_chart(df_trend, "Tahun", "IPM")
            st.altair_chart(chart_trend, use_container_width=True)
            
        with c2:
            st.markdown(f"##### 🏆 Top 5 Wilayah ({latest_year})")
            st.dataframe(df_latest.nlargest(5, "IPM")[["Cakupan", "IPM"]].set_index("Cakupan"), use_container_width=True)
            st.markdown(f"##### ⚠️ Bottom 5 Wilayah ({latest_year})")
            st.dataframe(df_latest.nsmallest(5, "IPM")[["Cakupan", "IPM"]].set_index("Cakupan"), use_container_width=True)

        st.markdown("---")
        st.subheader("🔍 Komparasi Daerah")
        daerah_list = sorted(df_hist["Cakupan"].unique())
        selected_daerah = st.multiselect("Pilih Daerah:", daerah_list, default=daerah_list[:2] if len(daerah_list)>1 else None)
        
        if selected_daerah:
            df_comp = df_hist[df_hist["Cakupan"].isin(selected_daerah)]
            # Gunakan chart Altair juga disini agar konsisten dan rapi
            chart_comp = alt.Chart(df_comp).mark_line(point=True).encode(
                x=alt.X('Tahun:Q', axis=alt.Axis(format='d', tickMinStep=1)), # Format tahun diperbaiki
                y=alt.Y('IPM:Q', scale=alt.Scale(zero=False)),
                color='Cakupan:N',
                tooltip=['Cakupan', alt.Tooltip('Tahun', format='d'), 'IPM']
            ).properties(height=350).interactive()
            st.altair_chart(chart_comp, use_container_width=True)

# -------------------------------------------------------
# TAB 2: FORECASTING
# -------------------------------------------------------
with tab2:
    st.subheader("🔮 Proyeksi IPM Masa Depan")
    if df_hist is not None and model is not None:
        col_f1, col_f2 = st.columns([1, 3])
        with col_f1:
            daerah_fc = st.selectbox("Pilih Wilayah:", sorted(df_hist["Cakupan"].unique()))
            horizon = st.slider("Jangka Waktu:", 1, 10, 5)
            btn_fc = st.button("Generate Forecast")
            
        with col_f2:
            if btn_fc:
                df_d = df_hist[df_hist["Cakupan"] == daerah_fc].sort_values("Tahun")
                if len(df_d) < 2:
                    st.error("Data historis kurang.")
                else:
                    features = ["UHH", "HLS", "RLS", "Pengeluaran"]
                    growth = df_d[features].diff().mean()
                    last_row = df_d.iloc[-1]
                    
                    future_data = []
                    curr = last_row[features].to_dict()
                    
                    for i in range(1, horizon + 1):
                        for f in features: curr[f] += growth[f]
                        row = curr.copy()
                        row["Tahun"] = int(last_row["Tahun"]) + i
                        input_df = pd.DataFrame([row])[feature_names] if feature_names else pd.DataFrame([row])
                        row["IPM"] = model.predict(input_df)[0]
                        row["Tipe"] = "Forecast"
                        future_data.append(row)
                    
                    df_future = pd.DataFrame(future_data)
                    df_hist_chart = df_d[["Tahun", "IPM"]].copy()
                    df_hist_chart["Tipe"] = "Aktual"
                    df_final = pd.concat([df_hist_chart, df_future[["Tahun", "IPM", "Tipe"]]], ignore_index=True)
                    
                    st.markdown(f"##### Hasil Forecast: {daerah_fc}")
                    
                    # Gunakan fungsi chart yang sudah diperbaiki
                    fc_chart = create_labeled_line_chart(df_final, "Tahun", "IPM", color_col="Tipe")
                    st.altair_chart(fc_chart, use_container_width=True)
                    
                    st.dataframe(df_future[["Tahun", "IPM", "UHH", "HLS", "RLS", "Pengeluaran"]].style.format("{:.2f}"), use_container_width=True)
                    
                    csv_forecast = df_future.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="💾 Download Data Forecast (CSV)",
                        data=csv_forecast,
                        file_name=f"forecast_{daerah_fc}_{horizon}tahun.csv",
                        mime="text/csv",
                        key="download_forecast"
                    )

# -------------------------------------------------------
# TAB 3: BULK UPLOAD
# -------------------------------------------------------
with tab3:
    st.subheader("📤 Prediksi Massal via CSV")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file and model:
        try:
            df_in = pd.read_csv(uploaded_file)
            missing = [c for c in feature_names if c not in df_in.columns]
            if not missing:
                df_in["IPM_Prediksi"] = model.predict(df_in[feature_names])
                st.success("✅ Prediksi Selesai")
                st.dataframe(df_in.head(), use_container_width=True)
                
                csv_res = df_in.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Download Hasil Prediksi (CSV)",
                    data=csv_res,
                    file_name="hasil_prediksi_ipm_massal.csv",
                    mime="text/csv",
                    key="download_bulk"
                )
            else:
                st.error(f"Kolom hilang: {missing}")
        except Exception as e: st.error(f"Error: {e}")
