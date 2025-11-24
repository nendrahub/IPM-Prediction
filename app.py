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
# TAB 2 – FORECAST BERDASARKAN GROWTH (DRIFT)
# ======================================
with tab2:
    st.subheader("🔮 Forecast IPM per Daerah (Metode Drift)")

    if df_hist is None:
        st.warning("File data_ipm.csv tidak ditemukan. Forecast tidak dapat dihitung.")
    else:
        daerah_list = sorted(df_hist["Cakupan"].unique().tolist())
        daerah2 = st.selectbox(
            "Pilih Provinsi/Daerah untuk Forecast",
            daerah_list,
            key="forecast_daerah"
        )

        # Ambil data per daerah & urutkan tahun
        df_d = df_hist[df_hist["Cakupan"] == daerah2].sort_values("Tahun").copy()

        if df_d["Tahun"].nunique() < 2:
            st.warning("Data tahun terlalu sedikit.")
        else:
            # --- 1. HITUNG GROWTH (DRIFT) ---
            # Secara matematis: Rata-rata selisih tahunan = Slope Drift Method
            df_d["UHH_diff"] = df_d["UHH"].diff()
            df_d["HLS_diff"] = df_d["HLS"].diff()
            df_d["RLS_diff"] = df_d["RLS"].diff()
            df_d["Pengeluaran_diff"] = df_d["Pengeluaran"].diff()

            growth = df_d[["UHH_diff", "HLS_diff", "RLS_diff", "Pengeluaran_diff"]].mean()

            last_row = df_d.tail(1).copy()
            last_year = int(last_row["Tahun"].iloc[0])

            horizon = st.slider("Horizon tahun forecast", 1, 10, 5)

            # --- 2. GENERATE FORECAST ---
            future_rows = []
            current = last_row.copy()

            for i in range(1, horizon + 1):
                new = current.copy()
                new["Tahun"] = last_year + i
                
                # Rumus Drift: Nilai Baru = Nilai Lama + Rata2 Growth
                new["UHH"] = new["UHH"] + growth["UHH_diff"]
                new["HLS"] = new["HLS"] + growth["HLS_diff"]
                new["RLS"] = new["RLS"] + growth["RLS_diff"]
                new["Pengeluaran"] = new["Pengeluaran"] + growth["Pengeluaran_diff"]

                # Prediksi IPM pakai Model ML
                X_new = new[feature_names]
                new["IPM (Forecast)"] = model.predict(X_new)[0]
                
                future_rows.append(new)
                current = new

            if future_rows:
                df_future = pd.concat(future_rows, ignore_index=True)

                st.write(f"**Forecast IPM {daerah2} untuk {horizon} tahun ke depan:**")
                
                # Tampilkan Tabel
                st.dataframe(
                    df_future[["Tahun", "UHH", "HLS", "RLS", "Pengeluaran", "IPM (Forecast)"]],
                    use_container_width=True
                )

                # --- 3. VISUALISASI CHART (Updated Style) ---
                
                # Siapkan Data Aktual (IPM)
                df_plot_hist = df_d[["Tahun", "IPM"]].copy()
                df_plot_hist["Jenis_Data"] = "Aktual"
                df_plot_hist = df_plot_hist.rename(columns={"IPM": "Nilai IPM"})

                # Siapkan Data Forecast (IPM Forecast)
                df_plot_future = df_future[["Tahun", "IPM (Forecast)"]].copy()
                df_plot_future["Jenis_Data"] = "Forecast (Drift)"
                df_plot_future = df_plot_future.rename(columns={"IPM (Forecast)": "Nilai IPM"})

                # Gabung jadi satu DataFrame panjang (Long Format)
                df_plot_combined = pd.concat([df_plot_hist, df_plot_future], ignore_index=True)

                # Definisi Warna: Biru (Aktual), Oranye (Forecast)
                color_scale = alt.Scale(
                    domain=['Aktual', 'Forecast (Drift)'],
                    range=['#1f77b4', '#ff7f0e']
                )

                chart = alt.Chart(df_plot_combined).mark_line(
                    point=alt.OverlayMarkDef(filled=True, size=60)
                ).encode(
                    x=alt.X('Tahun', axis=alt.Axis(format='d', title='Tahun')), # Format 'd' agar 2022 (bukan 2,022)
                    y=alt.Y('Nilai IPM', scale=alt.Scale(zero=False), title='Nilai IPM'),
                    color=alt.Color('Jenis_Data', scale=color_scale, legend=alt.Legend(title="Keterangan")),
                    tooltip=['Tahun', 'Jenis_Data', alt.Tooltip('Nilai IPM', format='.2f')]
                ).interactive()

                st.altair_chart(chart, use_container_width=True)

                # --- 4. DOWNLOAD BUTTON ---
                csv_future = df_future[
                    ["Tahun", "IPM (Forecast)", "UHH", "HLS", "RLS", "Pengeluaran"]
                ].to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="💾 Download hasil forecast (CSV)",
                    data=csv_future,
                    file_name=f"forecast_ipm_{daerah2.replace(' ', '_')}.csv",
                    mime="text/csv"
                )

import altair as alt # Pastikan library ini terimport

# ======================================
# FUNGSI DRIFT METHOD
# (Letakkan fungsi ini di luar 'with tab3' atau di bagian paling atas script)
# ======================================
def forecast_drift(series, target_year):
    series = series.dropna()
    if len(series) < 2:
        return series.iloc[-1] if len(series) > 0 else 0
    
    y_last = series.iloc[-1]
    y_first = series.iloc[0]
    T = len(series)
    slope = (y_last - y_first) / (T - 1)
    
    last_year_idx = series.index[-1]
    h = target_year - last_year_idx
    
    if h <= 0: return y_last
    return y_last + (h * slope)

# ======================================
# TAB 3 – FINAL CODE (CUSTOM CHART VISUALIZATION)
# ======================================
with tab3:
    st.subheader("📤 Upload Data & Forecasting (Clean Output)")
    
    st.write("Sistem akan menggabungkan Data Aktual dan Forecasting ke dalam satu kolom IPM yang rapi.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            # 1. BACA FILE
            df_input = pd.read_csv(uploaded_file, decimal=",", thousands=".")
            
            # 2. BERSIHKAN NAMA KOLOM
            df_input.columns = df_input.columns.str.strip()
            
            rename_map = {
                'tahun': 'Tahun', 'TAHUN': 'Tahun',
                'uhh': 'UHH', 'Uhh': 'UHH',
                'hls': 'HLS', 'Hls': 'HLS',
                'rls': 'RLS', 'Rls': 'RLS',
                'pengeluaran': 'Pengeluaran', 'PENGELUARAN': 'Pengeluaran',
                'cakupan': 'Cakupan', 'CAKUPAN': 'Cakupan',
                'IPM': 'IPM',
                'Indeks Pembangunan Manusia': 'IPM'
            }
            df_input = df_input.rename(columns=rename_map)

            wajib = ['UHH', 'HLS', 'RLS', 'Pengeluaran', 'Tahun']
            if not all(col in df_input.columns for col in wajib):
                st.error(f"Kolom wajib tidak lengkap. Pastikan ada: {wajib}")
            else:
                # 3. PERSIAPAN DATA
                df_proc = df_input.copy()
                if 'Cakupan' not in df_proc.columns:
                    df_proc['Cakupan'] = 'Wilayah Upload'
                
                target_col_name = 'IPM'
                if target_col_name not in df_proc.columns:
                    df_proc[target_col_name] = np.nan

                df_proc = df_proc.sort_values(by=['Cakupan', 'Tahun'])

                # 4. LOOP FORECASTING
                all_data = []
                regions = df_proc['Cakupan'].unique()
                drift_cols = ['UHH', 'HLS', 'RLS', 'Pengeluaran']
                target_year = 2030

                bar = st.progress(0, "Menghitung forecast...")
                
                for i, region in enumerate(regions):
                    df_region = df_proc[df_proc['Cakupan'] == region].copy()
                    # Label 'Aktual' nanti akan kita warnai Biru
                    df_region['Tipe'] = 'Aktual'
                    all_data.append(df_region)
                    
                    last_year = int(df_region['Tahun'].max())
                    df_indexed = df_region.set_index('Tahun')

                    if last_year < target_year:
                        for yr in range(last_year + 1, target_year + 1):
                            new_row = {
                                'Cakupan': region,
                                'Tahun': yr,
                                'Tipe': 'Forecast', # Nanti warna Oranye
                                target_col_name: np.nan
                            }
                            for col in drift_cols:
                                new_row[col] = forecast_drift(df_indexed[col], yr)
                            
                            all_data.append(pd.DataFrame([new_row]))
                    
                    bar.progress((i + 1) / len(regions))
                
                bar.empty()

                # 5. GABUNGKAN DATA
                df_final = pd.concat(all_data, ignore_index=True)

                # 6. HITUNG PREDIKSI IPM
                model_features = ['UHH', 'HLS', 'RLS', 'Pengeluaran', 'Tahun']
                try:
                    df_final['TEMP_IPM (Forecast)'] = model.predict(df_final[model_features])
                except ValueError:
                     if 'feature_names' in globals():
                         df_final['TEMP_IPM (Forecast)'] = model.predict(df_final[feature_names])
                     else:
                        df_final['TEMP_IPM (Forecast)'] = 0 

                # 7. LOGIKA GABUNGAN
                df_final[target_col_name] = df_final[target_col_name].fillna(df_final['TEMP_IPM (Forecast)'])
                df_final[target_col_name] = df_final[target_col_name].round(2)

                # 8. OUTPUT FINAL
                final_cols = ['Cakupan', 'UHH', 'HLS', 'RLS', 'Pengeluaran', target_col_name, 'Tahun', 'Tipe']
                final_cols = [c for c in final_cols if c in df_final.columns]
                df_display = df_final[final_cols]

                st.success("✅ Data berhasil diproses!")
                st.dataframe(df_display.tail(10), use_container_width=True)

                # =========================================================
                # VISUALISASI CUSTOM (ALTAIR)
                # =========================================================
                st.write("**Grafik Tren IPM (Biru: Data Aktual, Oranye: Forecast):**")
                
                # Definisi Warna: Aktual -> Biru, Forecast -> Oranye
                color_scale = alt.Scale(
                    domain=['Aktual', 'Forecast'],
                    range=['#1f77b4', '#ff7f0e']  # Hex code: Biru standar & Oranye standar
                )

                # Membuat Chart
                chart = alt.Chart(df_display).mark_line(
                    point=alt.OverlayMarkDef(filled=True, size=60) # Menambahkan titik kecil
                ).encode(
                    # Sumbu X: Format 'd' untuk integer (menghilangkan koma 2,022)
                    x=alt.X('Tahun', axis=alt.Axis(format='d', title='Tahun')),
                    
                    # Sumbu Y: IPM, scale zero=False agar grafik tidak mulai dari 0 (lebih fokus)
                    y=alt.Y(target_col_name, scale=alt.Scale(zero=False), title='Nilai IPM'),
                    
                    # Warna berdasarkan Jenis Data
                    color=alt.Color('Tipe', scale=color_scale, legend=alt.Legend(title="Keterangan")),
                    
                    # Detail agar jika ada banyak wilayah (Cakupan), garisnya tidak nyambung sembarangan
                    detail='Cakupan',
                    
                    # Tooltip saat hover mouse
                    tooltip=['Cakupan', 'Tahun', target_col_name, 'Tipe']
                ).interactive() # Bisa di-zoom/pan

                st.altair_chart(chart, use_container_width=True)

                # Download Button
                csv = df_display.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Download CSV Final",
                    data=csv,
                    file_name="Hasil_Forecast_Final.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Terjadi error: {e}")








