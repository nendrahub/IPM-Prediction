import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import altair as alt  # Ditambahkan untuk kustomisasi chart

# ============================
# LOAD MODEL & DATA
# ============================

@st.cache_resource
def load_model():
    loaded = joblib.load("model_ipm_gradientboosting.joblib")
    return loaded["model"], loaded["features"]

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
# HEADER
# ============================

st.title("📈 Prediksi & Forecasting IPM Indonesia")

st.write(
    "Aplikasi ini menggunakan model **Gradient Boosting** untuk memprediksi "
    "Indeks Pembangunan Manusia (IPM) berdasarkan komponen penyusunnya: "
    "**UHH, HLS, RLS, Pengeluaran per Kapita, dan Tahun**."
)

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

with st.sidebar.expander("ℹ️ Keterangan Singkatan Komponen"):
    st.markdown(
        """
        **UHH** : Umur Harapan Hidup (tahun)  
        **HLS** : Harapan Lama Sekolah (tahun)  
        **RLS** : Rata-rata Lama Sekolah (tahun)  
        **Pengeluaran** : Pengeluaran per Kapita (Rp/tahun, harga konstan)  
        **IPM** : Indeks Pembangunan Manusia  
        """
    )

# ============================
# TAB UTAMA
# ============================

tab1, tab2, tab3 = st.tabs([
    "📊 Visualisasi IPM & Ringkasan Tahunan",
    "🔮 Forecast IPM per Daerah",
    "📤 Upload Data & Prediksi Massal"
])

# ======================================
# TAB 1 – VISUALISASI IPM LEBIH BERMAKNA
# ======================================
with tab1:
    st.subheader("📊 Ringkasan IPM Nasional & Per Daerah")

    if df_hist is None:
        st.warning("File data_ipm.csv tidak ditemukan. Visualisasi historis tidak dapat ditampilkan.")
    else:
        # --------- BAGIAN 1: Tren nasional ---------
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Tren Rata-rata IPM Nasional")
            df_nat = (
                df_hist.groupby("Tahun")["IPM"]
                .agg(["mean", "min", "max"])
                .reset_index()
                .rename(columns={"mean": "IPM_Rata2", "min": "IPM_Min", "max": "IPM_Maks"})
            )

            # UPDATED VISUALIZATION: Altair dengan Line, Point, dan Label
            chart_nat = alt.Chart(df_nat).encode(
                x=alt.X('Tahun:O', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('IPM_Rata2:Q', scale=alt.Scale(zero=False), title='IPM Rata-rata'),
                tooltip=['Tahun', alt.Tooltip('IPM_Rata2', format='.2f')]
            )
            line_nat = chart_nat.mark_line()
            point_nat = chart_nat.mark_point(filled=True, size=60)
            text_nat = chart_nat.mark_text(align='left', dx=5, dy=-10).encode(
                text=alt.Text('IPM_Rata2:Q', format='.2f')
            )
            
            st.altair_chart((line_nat + point_nat + text_nat).interactive(), use_container_width=True)
            
            st.caption(
                "Garis di atas menunjukkan perkembangan **IPM rata-rata Indonesia** per tahun."
            )

            st.write("📌 Ringkasan statistik per tahun:")
            st.dataframe(df_nat, use_container_width=True)

        # --------- BAGIAN 2: Ringkasan per tahun (top/bottom) ---------
        with col2:
            st.markdown("### Ringkasan Daerah per Tahun")

            tahun_ringkas = st.selectbox(
                "Pilih tahun untuk melihat peringkat IPM",
                sorted(df_hist["Tahun"].unique()),
            )

            df_year = df_hist[df_hist["Tahun"] == tahun_ringkas].copy()
            avg_ipm = df_year["IPM"].mean()
            max_ipm = df_year["IPM"].max()
            min_ipm = df_year["IPM"].min()

            c1, c2, c3 = st.columns(3)
            c1.metric("Rata-rata IPM", f"{avg_ipm:.2f}")
            c2.metric("IPM Tertinggi", f"{max_ipm:.2f}")
            c3.metric("IPM Terendah", f"{min_ipm:.2f}")

            df_rank = df_year.sort_values("IPM", ascending=False)

            st.markdown("**Top 5 daerah dengan IPM tertinggi:**")
            st.dataframe(df_rank.head(5)[["Cakupan", "IPM"]])

            st.markdown("**Top 5 daerah dengan IPM terendah:**")
            st.dataframe(df_rank.tail(5)[["Cakupan", "IPM"]])

        st.markdown("---")
        # --------- BAGIAN 3: Perbandingan beberapa daerah ---------
        st.markdown("### Perbandingan Tren IPM Beberapa Daerah")

        daerah_list = sorted(df_hist["Cakupan"].unique().tolist())
        default_selection = daerah_list[:3] if len(daerah_list) >= 3 else daerah_list

        daerah_multi = st.multiselect(
            "Pilih 1–5 daerah untuk dibandingkan tren IPM-nya:",
            daerah_list,
            default=default_selection,
        )

        if daerah_multi:
            df_sel = (
                df_hist[df_hist["Cakupan"].isin(daerah_multi)]
                .sort_values(["Cakupan", "Tahun"])
            )

            # UPDATED VISUALIZATION: Altair dengan Line, Point, dan Label
            chart_sel = alt.Chart(df_sel).encode(
                x=alt.X('Tahun:O', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('IPM:Q', scale=alt.Scale(zero=False)),
                color='Cakupan:N',
                tooltip=['Tahun', 'Cakupan', alt.Tooltip('IPM', format='.2f')]
            )
            line_sel = chart_sel.mark_line()
            point_sel = chart_sel.mark_point(filled=True, size=60)
            text_sel = chart_sel.mark_text(align='left', dx=5, dy=-10).encode(
                text=alt.Text('IPM:Q', format='.2f')
            )

            st.altair_chart((line_sel + point_sel + text_sel).interactive(), use_container_width=True)

        else:
            st.info("Pilih minimal satu daerah untuk melihat perbandingan tren IPM.")

# ======================================
# TAB 2 – FORECAST BERDASARKAN GROWTH
# ======================================
with tab2:
    st.subheader("🔮 Forecast IPM per Daerah")

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
                ).sort_values("Tahun")

                # UPDATED VISUALIZATION: Melt data untuk Altair agar bisa memberi warna berbeda dan label
                df_plot_melt = df_plot.melt("Tahun", var_name="Jenis", value_name="Nilai_IPM").dropna()

                chart_fore = alt.Chart(df_plot_melt).encode(
                    x=alt.X('Tahun:O', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('Nilai_IPM:Q', scale=alt.Scale(zero=False)),
                    color='Jenis:N',
                    tooltip=['Tahun', 'Jenis', alt.Tooltip('Nilai_IPM', format='.2f')]
                )
                line_fore = chart_fore.mark_line()
                point_fore = chart_fore.mark_point(filled=True, size=60)
                text_fore = chart_fore.mark_text(align='left', dx=5, dy=-10).encode(
                    text=alt.Text('Nilai_IPM:Q', format='.2f')
                )

                st.altair_chart((line_fore + point_fore + text_fore).interactive(), use_container_width=True)

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
# FUNGSI DRIFT METHOD
# ======================================
def forecast_drift(series, target_year):
    """
    Memprediksi nilai masa depan menggunakan Drift Method.
    series: pandas Series (data historis urut tahun)
    target_year: int (tahun tujuan prediksi)
    """
    # Pastikan data tidak kosong dan minimal ada 2 titik data untuk menghitung slope
    if len(series) < 2:
        return series.iloc[-1] # Jika data cuma 1, gunakan nilai terakhir (Naïve)
    
    y_last = series.iloc[-1] # Nilai tahun terakhir
    y_first = series.iloc[0] # Nilai tahun pertama
    T = len(series)          # Jumlah data historis
    
    # Hitung rata-rata perubahan (slope)
    slope = (y_last - y_first) / (T - 1)
    
    # Hitung h (berapa tahun ke depan)
    # Kita asumsikan index series adalah Tahun
    last_year_idx = series.index[-1]
    h = target_year - last_year_idx
    
    if h <= 0:
        return y_last
        
    return y_last + (h * slope)

# ======================================
# TAB 3 – UPLOAD FILE & PREDIKSI MASSAL
# ======================================
with tab3:
    st.subheader("📤 Upload Data, Prediksi & Forecasting IPM (s.d. 2030)")

    st.write(
        "Fitur ini untuk **upload file CSV** komponen IPM. Sistem akan memprediksi IPM data saat ini "
        "dan melakukan **Forecasting (Drift Method)** untuk komponen UHH, HLS, RLS, & Pengeluaran hingga tahun 2030."
    )

    st.markdown(
        """
        **Format kolom yang diharapkan:**
        - `UHH`, `HLS`, `RLS`, `Pengeluaran`  
        - `Tahun`  
        - `Cakupan` (Opsional: Nama Provinsi/Kabupaten. Jika tidak ada, dianggap 1 wilayah).
        """
    )

    uploaded_file = st.file_uploader(
        "Upload file CSV komponen IPM",
        type=["csv"],
        help="Pastikan kolom minimal berisi: UHH, HLS, RLS, Pengeluaran, Tahun"
    )

    if uploaded_file is not None:
        try:
            # 1. Load Data
            df_input = pd.read_csv(
                uploaded_file,
                decimal=",",      
                thousands="."     
            )
            
            # Cek kolom wajib
            required_cols = feature_names + ['Tahun']
            missing = [col for col in required_cols if col not in df_input.columns]
            
            if missing:
                st.error(f"Kolom berikut tidak ditemukan di file: {missing}")
            else:
                # 2. Persiapan Data
                df_proc = df_input.copy()
                
                # Jika kolom Cakupan tidak ada, buat dummy agar loop tetap berjalan
                has_cakupan = 'Cakupan' in df_proc.columns
                if not has_cakupan:
                    df_proc['Cakupan'] = 'Wilayah Upload'

                # Pastikan urut berdasarkan Cakupan dan Tahun
                df_proc = df_proc.sort_values(by=['Cakupan', 'Tahun'])

                # List untuk menampung data historis + forecast
                all_data = []

                # 3. Loop per Wilayah (Cakupan) untuk Forecasting
                regions = df_proc['Cakupan'].unique()
                
                target_forecast_year = 2030
                
                forecast_bar = st.progress(0, text="Sedang menghitung forecast...")

                for i, region in enumerate(regions):
                    # Ambil data per wilayah
                    df_region = df_proc[df_proc['Cakupan'] == region].copy()
                    
                    # Set Tahun sebagai index untuk mempermudah fungsi drift
                    df_region_indexed = df_region.set_index('Tahun')
                    
                    last_year = df_region['Tahun'].max()
                    
                    # Tambahkan data historis ke list
                    # Kita tandai ini data aktual
                    df_region['Jenis_Data'] = 'Aktual'
                    all_data.append(df_region)
                    
                    # Lakukan forecasting jika last_year < 2030
                    if last_year < target_forecast_year:
                        future_years = range(last_year + 1, target_forecast_year + 1)
                        
                        for year in future_years:
                            new_row = {'Cakupan': region, 'Tahun': year, 'Jenis_Data': 'Forecast (Drift)'}
                            
                            # Hitung forecast untuk setiap fitur (UHH, HLS, dll)
                            for col in feature_names:
                                val = forecast_drift(df_region_indexed[col], year)
                                new_row[col] = val
                            
                            # Ubah dictionary ke DataFrame 1 baris
                            df_future_row = pd.DataFrame([new_row])
                            all_data.append(df_future_row)
                    
                    # Update progress bar
                    forecast_bar.progress((i + 1) / len(regions), text=f"Memproses wilayah: {region}")

                forecast_bar.empty()

                # 4. Gabungkan Kembali Data
                df_final = pd.concat(all_data, ignore_index=True)
                
                # Jika tadi kita buat dummy cakupan, hapus lagi jika user tidak upload kolom Cakupan
                if not has_cakupan:
                    df_final = df_final.drop(columns=['Cakupan'])

                # 5. Prediksi Nilai IPM (untuk data Aktual + Forecast)
                # Pastikan urutan kolom sesuai dengan yang diharapkan model
                df_final["IPM_Prediksi"] = model.predict(df_final[feature_names])

                # 6. Tampilkan Hasil
                st.success(f"Berhasil memproses data & forecasting hingga tahun {target_forecast_year}!")
                
                st.subheader("📊 Preview Hasil (Data Aktual & Forecast)")
                st.dataframe(df_final.tail(10), use_container_width=True) # Tampilkan 10 data terakhir (forecast)

                # Visualisasi Sederhana (Line Chart)
                st.write("**Grafik Tren IPM (Aktual & Prediksi):**")
                chart_data = df_final.set_index('Tahun')
                if has_cakupan:
                    st.line_chart(df_final, x='Tahun', y='IPM_Prediksi', color='Cakupan')
                else:
                    st.line_chart(chart_data['IPM_Prediksi'])

                # 7. Download Button
                csv_pred = df_final.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="💾 Download Hasil Lengkap (s.d. 2030)",
                    data=csv_pred,
                    file_name="hasil_forecast_ipm_2030.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Terjadi error: {e}")






