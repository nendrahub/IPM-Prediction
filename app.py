import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import altair as alt 

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
    /* Styling tombol download agar lebih menarik */
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
# FUNGSI PEMBUAT CHART DENGAN LABEL
# ============================
def create_labeled_line_chart(data, x_col, y_col, color_col=None, title=None):
    # Base Chart
    base = alt.Chart(data).encode(
        x=alt.X(f'{x_col}:O', axis=alt.Axis(labelAngle=0)), 
        y=alt.Y(f'{y_col}:Q', scale=alt.Scale(zero=False)), 
        tooltip=[x_col, alt.Tooltip(y_col, format=".2f")]
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
            df_pivot = df_comp.pivot(index="Tahun", columns="Cakupan", values="IPM")
            st.line_chart(df_pivot, height=350)

# ======================================
# TAB 2 – FORECAST BERDASARKAN GROWTH
# ======================================
with tab2:
    st.subheader("🔮 Forecast IPM dengan Growth Rata-rata per Daerah")

    if df_hist is None:
        st.warning("File data_ipm.csv tidak ditemukan. Forecast tidak dapat dihitung.")
    else:
        daerah_list = sorted(df_hist["Cakupan"].unique().tolist())
        daerah2 = st.selectbox(
            "Pilih Provinsi/Daerah untuk Forecast",
            daerah_list,
            key="forecast_daerah"
        )

        df_d = df_hist[df_hist["Cakupan"] == daerah2].sort_values("Tahun").copy()

        if df_d["Tahun"].nunique() < 2:
            st.warning("Data tahun untuk daerah ini terlalu sedikit untuk menghitung growth rata-rata.")
        else:
            # Hitung growth sederhana
            df_d["UHH_diff"] = df_d["UHH"].diff()
            df_d["HLS_diff"] = df_d["HLS"].diff()
            df_d["RLS_diff"] = df_d["RLS"].diff()
            df_d["Pengeluaran_diff"] = df_d["Pengeluaran"].diff()

            growth = df_d[["UHH_diff", "HLS_diff", "RLS_diff", "Pengeluaran_diff"]].mean()

            last_row = df_d.tail(1).copy()
            last_year = int(last_row["Tahun"].iloc[0])

            horizon = st.slider("Horizon tahun forecast", 1, 10, 5)

            future_rows = []
            current = last_row.copy()

            for i in range(1, horizon + 1):
                new = current.copy()
                new["Tahun"] = last_year + i
                new["UHH"] = new["UHH"] + growth["UHH_diff"]
                new["HLS"] = new["HLS"] + growth["HLS_diff"]
                new["RLS"] = new["RLS"] + growth["RLS_diff"]
                new["Pengeluaran"] = new["Pengeluaran"] + growth["Pengeluaran_diff"]

                X_new = new[feature_names]
                new["IPM_Prediksi"] = model.predict(X_new)[0]
                future_rows.append(new)

                current = new

            if future_rows:
                df_future = pd.concat(future_rows, ignore_index=True)

                st.write(f"**Forecast IPM {daerah2} untuk {horizon} tahun ke depan:**")
                st.dataframe(
                    df_future[["Tahun", "IPM_Prediksi", "UHH", "HLS", "RLS", "Pengeluaran"]],
                    use_container_width=True
                )

                # Gabungkan historis + forecast untuk plot
                df_plot_hist = df_d[["Tahun", "IPM"]].rename(columns={"IPM": "IPM_Aktual"})
                df_plot_future = df_future[["Tahun", "IPM_Prediksi"]]

                df_plot = pd.merge(
                    df_plot_hist,
                    df_plot_future,
                    on="Tahun",
                    how="outer"
                ).set_index("Tahun").sort_index()

                st.line_chart(df_plot, height=350)

                # 🔽 Tombol download hasil forecast
                csv_future = df_future[
                    ["Tahun", "IPM_Prediksi", "UHH", "HLS", "RLS", "Pengeluaran"]
                ].to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="💾 Download hasil forecast (CSV)",
                    data=csv_future,
                    file_name=f"forecast_ipm_{daerah2.replace(' ', '_')}.csv",
                    mime="text/csv"
                )

# ======================================
# TAB 3 – UPLOAD FILE & PREDIKSI MASSAL
# ======================================
with tab3:
    st.subheader("📤 Upload Data Komponen & Prediksi Massal IPM")

    st.write(
        "Fitur ini untuk **upload file CSV** berisi komponen IPM (dummy data atau data aktual), "
        "lalu sistem akan menghitung kolom **IPM_Prediksi** secara otomatis."
    )

    st.markdown(
        """
        **Format kolom yang diharapkan (header CSV):**
        - `UHH`  
        - `HLS`  
        - `RLS`  
        - `Pengeluaran`  
        - `Tahun`  
        - (opsional) `Cakupan` – nama provinsi/daerah
        """
    )

    uploaded_file = st.file_uploader(
        "Upload file CSV komponen IPM",
        type=["csv"],
        help="Pastikan kolom minimal berisi: UHH, HLS, RLS, Pengeluaran, Tahun"
    )

    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)

            missing = [col for col in feature_names if col not in df_input.columns]
            if missing:
                st.error(f"Kolom berikut tidak ditemukan di file: {missing}")
            else:
                df_pred = df_input.copy()
                df_pred["IPM_Prediksi"] = model.predict(df_pred[feature_names])

                st.success("Prediksi IPM berhasil dihitung.")
                st.write("**Contoh hasil (5 baris pertama):**")
                st.dataframe(df_pred.head(), use_container_width=True)

                # 🔽 Tombol download hasil prediksi
                csv_pred = df_pred.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="💾 Download hasil prediksi (CSV)",
                    data=csv_pred,
                    file_name="hasil_prediksi_ipm.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Terjadi error saat membaca file: {e}")
