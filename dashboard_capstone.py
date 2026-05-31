import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os
import glob
import zipfile
import cv2


# ─────────────────────────────────────────────
# DOWNLOAD DATASET DARI GOOGLE DRIVE
# ─────────────────────────────────────────────
@st.cache_resource
def download_dataset():
    import gdown
    ZIP_ID    = "1Jri1GjlzKFqqFl9BasDS4YSv20tCw99T"
    ZIP_PATH  = "dataset.zip"
    EXTRACT_TO = "dataset_rice"

    if os.path.exists(EXTRACT_TO):
        return EXTRACT_TO


    gdown.download(id=ZIP_ID, output=ZIP_PATH, quiet=False)

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_TO)

    os.remove(ZIP_PATH)
    return EXTRACT_TO

with st.spinner("⏳ Memuat dataset..."):
    BASE_PATH = download_dataset()

# ─────────────────────────────────────────────
# KONFIGURASI KELAS
# ─────────────────────────────────────────────
kelas_folder = ["Blast", "BrownSpot", "Healthy", "Tungro", "Unknown"]
kelas_label  = ["Blast", "Brown Spot", "Healthy", "Tungro", "Unknown"]
folder_to_label = dict(zip(kelas_folder, kelas_label))
label_to_folder = dict(zip(kelas_label, kelas_folder))
WARNA_KELAS = {
    "Blast":      "#FF6B2B",
    "Brown Spot": "#8B4513",
    "Healthy":    "#4CAF50",
    "Tungro":     "#FFC107",
    "Unknown": "#9E9E9E"
}
warna_kelas = [WARNA_KELAS[l] for l in kelas_label]

# Data asli sebelum augmentasi
data_train_before = {
    "Blast":      1559,
    "Brown Spot": 1714,
    "Healthy":    1804,
    "Tungro":     1783,
    "Unknown":    2146
}
data_test_before = {
    "Blast":      390,
    "Brown Spot": 429,
    "Healthy":    452,
    "Tungro":     446,
    "Unknown":    537
}
# Data train setelah augmentasi
data_train_after = {
    "Blast":      2000,
    "Brown Spot": 2000,
    "Healthy":    2000,
    "Tungro":     2000,
    "Unknown":    2000
}
data_test_after = data_test_before  

# Untuk hitung data fallback
@st.cache_data
def hitung_data(base_path):
    dt = {}
    dte = {}
    for folder, label in folder_to_label.items():
        pt = os.path.join(base_path, "train", folder)
        pe = os.path.join(base_path, "test",  folder)
        dt[label]  = len(glob.glob(os.path.join(pt, "*.jpg"))) if os.path.exists(pt) else data_train_before[label]
        dte[label] = len(glob.glob(os.path.join(pe, "*.jpg"))) if os.path.exists(pe) else data_test_before[label]
    return dt, dte

data_train, data_test = hitung_data(BASE_PATH)

# ─────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RiceCare AI - Exploratory Data Analysis",
    page_icon="🌾",
    layout="wide"
)

st.markdown("""
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: bold;
            color: #2e7d32;
            text-align: center;
        }
        .sub-title {
            font-size: 1rem;
            color: #555;
            text-align: center;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🌾 RiceCare AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Exploratory Data Analysis | Coding Camp 2026 × DBS Foundation | Tim CC26-PSU169</div>', unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://news.cgtn.com/news/2023-08-09/Chinese-researchers-discover-key-gene-for-rice-yield-increase-1m7pYKIw65O/img/a817995040884e35ac6a878ddec3fed8/a817995040884e35ac6a878ddec3fed8.jpeg", use_container_width=True)
    st.markdown("## 🌾 RiceCare AI")
    st.divider()
    halaman = st.radio(
        "Pilih Halaman:",
        ["📋 Overview & Data Dictionary", "📊 Explore & Explain Data", "🖼️ Image Samples"]
    )
    st.divider()
    st.caption("Data Scientist Team: Nisa Nuraini & Rifa Agnia")

# ─────────────────────────────────────────────
# Halaman 1: OVERVIEW & DATA DICTIONARY
# ─────────────────────────────────────────────
if halaman == "📋 Overview & Data Dictionary":
    st.subheader("📋 Project Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Kelas", "5")
    col2.metric("Train Data", sum(data_train_after.values()))
    col3.metric("Test Data",  sum(data_test_before.values()))
    col4.metric("Total Gambar", sum(data_train_after.values()) + sum(data_test_before.values()))

    # sumber data yang digunakan
    with st.expander("📁 Sumber Dataset"):
        st.markdown("""
        | No | Nama Dataset | Link |
        |---|---|---|
        | 1 | Rice Disease Dataset | [Kaggle](https://www.kaggle.com/datasets/anshulm257/rice-disease-dataset) | 
        | 2 | Rice Leaf Disease Dataset | [Kaggle](https://www.kaggle.com/datasets/soni535/rice-leaf-bacterial-and-fungal-disease) | 
        | 3 | Rice Leaf Diseases Dataset | [Kaggle](https://www.kaggle.com/datasets/raihan150146/rice-leaf-diseases-dataset) |
        | 4 | Zambali Rice Dataset (Public V1) | [Kaggle](https://www.kaggle.com/datasets/gettingintoml/zambali-rice-dataset-v3-1) | 
        | 5 | Rice Leaf Disease Detection Dataset | [Kaggle](https://www.kaggle.com/datasets/sham69/rice-leaf-disease-detection-dataset) | 
        | 6 | Dataset Of Leaf Diseases And Pests Of Agricultural Plants | [Kaggle](https://www.kaggle.com/datasets/dellcat/datasetofleafdiseasesandpestsofagriculturalplants) | 
        | 7 | FIRE Dataset | [Kaggle](https://www.kaggle.com/datasets/phylake1337/fire-dataset) | 
        | 8 | PlantVillage Dataset | [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease) |  
        | 9 | image captioning dataset, random images | [Kaggle](https://www.kaggle.com/datasets/shamsaddin97/image-captioning-dataset-random-images) |
        | 10 | Vehicle dataset - Images & Segmentation | [Kaggle](https://www.kaggle.com/datasets/trainingdatapro/car-masks) | 
                    """)

    st.divider()
    st.subheader("📖 Data Dictionary")
    dict_data = {
        "Atribut": ["image", "label", "split", "width", "height", "color_mode", "format"],
        "Tipe Data": ["Array", "String", "String", "Integer", "Integer", "String", "String"],
        "Deskripsi": [
            "Data piksel gambar daun padi yang menjadi input utama model",
            "Kelas kondisi daun padi yang menjadi target prediksi model",
            "Pembagian dataset untuk keperluan pelatihan dan evaluasi model",
            "Lebar dimensi gambar setelah standarisasi",
            "Tinggi dimensi gambar setelah standarisasi",
            "Mode warna gambar setelah proses konversi",
            "Format file gambar setelah proses konversi"
        ],
        "Spesifikasi": ["Nilai piksel (0-225), 3 channel RGB", "Blast, Brown Spot, Tungro, Healthy, Unknown", "Train, Test", "224 piksel", "224 piksel", "RGB 3 channel", "JPEG"]
    }
    st.dataframe(pd.DataFrame(dict_data), use_container_width=True)

    st.divider()
    st.subheader("🍂 Penjelasan Kelas Penyakit pada Daun Padi")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown("**🟤 Brown Spot**")
        st.write("Bercak coklat pada daun akibat jamur Helminthosporium oryzae. Menyebabkan biji padi tidak terisi penuh.")
    with col2:
        st.markdown("**🟠 Blast**")
        st.write("Penyakit jamur Magnaporthe oryzae yang menyerang daun. Salah satu penyakit paling merusak.")
    with col3:
        st.markdown("**🟡 Tungro**")
        st.write("Disebabkan oleh virus yang ditularkan wereng hijau. Daun menguning dan pertumbuhan terhambat.")
    with col4:
        st.markdown("**🟢 Healthy**")
        st.write("Daun padi dalam kondisi sehat, tidak menunjukkan gejala penyakit apapun.")
    with col5:
        st.markdown("**⚫ Unknown**")
        st.write("Kondisi daun yang tidak termasuk 4 kelas yang dikenali model. Muncul ketika tingkat kepercayaan prediksi di bawah threshold yang ditentukan.")

# ─────────────────────────────────────────────
# Halaman 2: VISUALISASI EDA
# ─────────────────────────────────────────────
elif halaman == "📊 Explore & Explain Data":
    st.subheader("📊 Visualisasi Distribusi Data")

    # ── SECTION 1: DISTRIBUSI DATASET ──
    st.markdown("### 🗂️ Distribusi Dataset")

    data_raw = {
    "Blast":      2587,
    "Brown Spot": 3293,
    "Healthy":    2597,
    "Tungro":     2352,
    "Unknown":    2683,
    }
    data_cleaned = {
    "Blast":      1949,
    "Brown Spot": 2143,
    "Healthy":    2256,
    "Tungro":     2229,
    "Unknown":    2683,
    }
    data_hapus = {k: data_raw[k] - data_cleaned[k] for k in data_raw}

    # Penjelasan di atas
    st.info("""
    **📦 Asal Usul Dataset:**

    Data dikumpulkan dari berbagai sumber di **Kaggle** melalui proses scraping dan penggabungan manual.
    Setelah terkumpul, dilakukan proses berikut:

    🔍 **Data Assessing & Cleaning:**
    - Pengecekan duplikat → **{} gambar duplikat dihapus**
    - Pengecekan file corrupted → **tidak ditemukan**

    ⚙️ **Preprocessing:**
    - Penyeragaman format gambar ke **JPG**
    - Konversi mode warna ke **RGB**
    - Resize gambar ke **224×224 piksel**
    """.format(sum(data_hapus.values())))

    # Dropdown
    pilihan_dist = st.selectbox(
        "Tampilkan kondisi:",
        options=["Sebelum Cleaning (Raw)", "Sesudah Cleaning"],
        index=0,
        key="selectbox_distribusi"
    )

    if pilihan_dist == "Sebelum Cleaning (Raw)":
        data_dipilih = data_raw
        judul_dist = "Distribusi Dataset — Sebelum Cleaning (Raw)"
        total_dipilih = sum(data_raw.values())
    else:
        data_dipilih = data_cleaned
        judul_dist = "Distribusi Dataset — Sesudah Cleaning"
        total_dipilih = sum(data_cleaned.values())
 
    # Metric row
    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    mc1.metric("Total Dataset", f"{total_dipilih:,}")
    mc2.metric("Blast",      f"{data_dipilih['Blast']:,}")
    mc3.metric("Brown Spot", f"{data_dipilih['Brown Spot']:,}")
    mc4.metric("Healthy",    f"{data_dipilih['Healthy']:,}")
    mc5.metric("Tungro",     f"{data_dipilih['Tungro']:,}")
    mc6.metric("Unknown",    f"{data_dipilih['Unknown']:,}")

    # Donut chart + insight
    col_donut, col_spacer = st.columns([1, 1])
    with col_donut:
        df_dist = pd.DataFrame(list(data_dipilih.items()), columns=["Kelas", "Jumlah"])
        fig_donut = px.pie(
            df_dist, values="Jumlah",
            names="Kelas",
            color="Kelas",
            color_discrete_map=WARNA_KELAS,
            hole=0.5,
            title=judul_dist
        )
        fig_donut.update_traces(textinfo="percent")
        fig_donut.update_layout(
            showlegend=True,
            margin=dict(t=40, b=10),
            annotations=[dict(
                text=f"<b>{total_dipilih:,}</b><br>gambar",
                x=0.5, y=0.5,
                font_size=14,
                showarrow=False
            )]
        )
        st.plotly_chart(fig_donut, use_container_width=True, key="donut_total")

    with col_spacer:
        if pilihan_dist == "Sebelum Cleaning (Raw)":
            st.info("""
            **🔍 Ringkasan Raw Data:**

            Data mentah hasil scraping dari Kaggle sebelum dilakukan proses cleaning.

            * 📈 **Brown Spot** menjadi kelas terbanyak dengan **3.293** gambar.
            * 📉 **Tungro** menjadi kelas paling sedikit dengan **2.352** gambar.

            Total terdapat **{}** gambar duplikat yang teridentifikasi saat proses assessing.
            """.format(sum(data_hapus.values())))
        else:
            st.info("""
            **✅ Ringkasan Data Setelah Cleaning:**

            Data setelah proses assessing & cleaning siap digunakan untuk pelatihan model.

            * 📈 **Unknown** menjadi kelas terbanyak dengan **2.683** gambar.
            * 📉 **Blast** menjadi kelas paling sedikit dengan **1.949** gambar.

            Dataset kemudian dibagi menjadi **80% train dan 20% test**. Khusus data train dilakukan
            **augmentasi data** untuk menyeimbangkan jumlah menjadi **2.000 gambar per kelas**.
            """)
    st.divider()

    # ── SECTION 2: BAR CHART PERBANDINGAN TRAIN VS TEST ──
    st.markdown("### 📊 Perbandingan Jumlah Data Train vs Test per Kelas")
    df_compare = pd.DataFrame({
        "Kelas": kelas_label,
        "Train": [data_train_before[l] for l in kelas_label],
        "Test":  [data_test_before[l]  for l in kelas_label]
    })
    fig_group = px.bar(
        df_compare.melt(id_vars="Kelas", var_name="Split", value_name="Jumlah"),
        x="Kelas", y="Jumlah", color="Split", barmode="group",
        color_discrete_sequence=["#42a5f5", "#ef5350"],
        text="Jumlah"
    )
    fig_group.update_traces(textposition="outside")
    fig_group.update_layout(yaxis_title="Jumlah Gambar", legend_title="Split")
    st.plotly_chart(fig_group, use_container_width=True, key="bar_train_test")

    # Insight kenapa rasio 80:20
    st.info("""
    **📊 Mengapa rasio Train:Test = 80:20?**

    Dataset dibagi dengan rasio **80% train dan 20% test** berdasarkan praktik umum dalam machine learning:

    - 🏋️ **80% Train** — Digunakan untuk melatih model agar dapat mengenali pola penyakit pada daun padi.
    - 🧪 **20% Test** — Digunakan untuk mengevaluasi performa model pada data yang **belum pernah dilihat** sebelumnya.

    Rasio ini dipilih karena dataset berukuran sedang (~10.000 gambar), sehingga 20% test sudah cukup representatif untuk mengukur generalisasi model.
    """)

    st.divider()

    # ── SECTION 3: BEFORE vs AFTER AUGMENTASI ──
    st.markdown("### ⚖️ Distribusi Data Train: Sebelum vs Sesudah Augmentasi")
    st.write("Augmentasi dilakukan hanya pada data train untuk menyeimbangkan jumlah tiap kelas menjadi **2.000 gambar per kelas**.")

    pilihan_aug = st.selectbox(
        "Tampilkan kondisi:",
        options=["Sebelum Augmentasi", "Sesudah Augmentasi"],
        index=0
    )
 
    if pilihan_aug == "Sebelum Augmentasi":
        df_aug = pd.DataFrame({
            "Kelas":  kelas_label,
            "Jumlah": [data_train_before[l] for l in kelas_label]
        })
        judul_aug = "Distribusi Data Train — Sebelum Augmentasi"
    else:
        df_aug = pd.DataFrame({
            "Kelas":  kelas_label,
            "Jumlah": [data_train_after[l] for l in kelas_label]
        })
        judul_aug = "Distribusi Data Train — Sesudah Augmentasi"
 
    fig_aug = px.bar(
        df_aug, x="Kelas", y="Jumlah",
        color="Kelas",
        color_discrete_map=WARNA_KELAS,
        text="Jumlah",
        title=judul_aug
    )
    fig_aug.update_traces(textposition="outside")
    fig_aug.update_layout(
        showlegend=False,
        yaxis_title="Jumlah Gambar",
        yaxis_range=[0, 2300]
    )
    st.plotly_chart(fig_aug, use_container_width=True, key="bar_augmentasi")

    #st.plotly_chart(fig_group, use_container_width=True)

    # Tabel ringkasan
    df_delta = pd.DataFrame({
        "Kelas":           kelas_label,
        "Sebelum (Train)": [data_train_before[l] for l in kelas_label],
        "Sesudah (Train)": [data_train_after[l]  for l in kelas_label],
        "Penambahan":      [data_train_after[l] - data_train_before[l] for l in kelas_label],
    })
    st.dataframe(df_delta, use_container_width=True, hide_index=True)

    with st.expander("💡 Mengapa augmentasi diperlukan?"):
        st.info("""
        **Distribusi data train sebelum augmentasi tidak seimbang:**
        - Kelas terbanyak (Healthy): **1.804** gambar
        - Kelas tersedikit (Blast): **1.559** gambar

        **Setelah augmentasi**, semua kelas train menjadi **2.000 gambar** (balanced).
        Teknik augmentasi yang digunakan: rotasi, flip horizontal/vertikal, brightness & contrast adjustment, dan blur.

        Kelas **Unknown** tidak diaugmentasi karena jumlah datanya sudah mencukupi (**2.160+ gambar**).

        Data test **tidak diaugmentasi** agar tetap merepresentasikan kondisi nyata di lapangan.
        """)

    st.divider()

    # ── SECTION 4: GAMBAR TIPIKAL PER KELAS ──
    st.markdown("### 🖼️ Gambar Tipikal per Kelas")
    st.write("Gambar asli yang paling merepresentasikan karakteristik tiap kelas (dipilih berdasarkan kedekatan ke rata-rata kelas)")

    @st.cache_data
    def cari_gambar_tipikal(base_path):
        hasil = {}
        for folder, label in folder_to_label.items():
            folder_path  = os.path.join(base_path, "train", folder)
            semua_gambar = glob.glob(os.path.join(folder_path, "*.jpg"))[:50]
            
            if semua_gambar:
                arrays = [np.array(Image.open(p).resize((224, 224)).convert("RGB")) for p in semua_gambar]
                mean_arr = np.mean(arrays, axis=0)
                jarak = [np.linalg.norm(arr - mean_arr) for arr in arrays]
                idx_tipikal = np.argmin(jarak)
                hasil[label] = arrays[idx_tipikal]
            else:
                hasil[label] = np.random.randint(80, 180, (224, 224, 3), dtype=np.uint8)
        
        return hasil

    with st.spinner("Mencari gambar tipikal per kelas..."):
        gambar_tipikal = cari_gambar_tipikal(BASE_PATH)

    cols = st.columns(len(kelas_label))
    for idx, label in enumerate(kelas_label):
        with cols[idx]:
            st.image(
                gambar_tipikal[label],
                caption=label,
                use_container_width=True
            )

    with st.expander("💡 Apa itu gambar tipikal?"):
        st.info("""
        Gambar tipikal adalah foto asli dari dataset yang paling mendekati rata-rata kelasnya.
        Berbeda dengan mean image yang menghasilkan gambar blur, gambar tipikal menampilkan
        foto nyata sehingga lebih mudah dipahami dan lebih representatif.
        """)
    
    @st.cache_data
    def hitung_rgb_per_kelas(base_path):
        hasil_rgb = {}
        for folder, label in folder_to_label.items():
            folder_path  = os.path.join(base_path, "train", folder)
            semua_gambar = glob.glob(os.path.join(folder_path, "*.jpg"))[:50]
            
            if semua_gambar:
                all_r, all_g, all_b = [], [], []
                for p in semua_gambar:
                    img = np.array(Image.open(p).resize((224, 224)).convert("RGB"))
                    all_r.append(img[:,:,0].mean())
                    all_g.append(img[:,:,1].mean())
                    all_b.append(img[:,:,2].mean())
                hasil_rgb[label] = {
                    "R": float(np.mean(all_r)),
                    "G": float(np.mean(all_g)),
                    "B": float(np.mean(all_b))
                }
            else:
                hasil_rgb[label] = {"R": 128.0, "G": 128.0, "B": 128.0}
        
        return hasil_rgb

    with st.spinner("Menghitung distribusi RGB..."):
        rgb_per_kelas = hitung_rgb_per_kelas(BASE_PATH)

    # ── SECTION 5: DISTRIBUSI RGB ──
    st.markdown("### 🎨 Distribusi Rata-rata Nilai RGB per Kelas")
    df_rgb = pd.DataFrame({
        "Kelas":     kelas_label,
        "R (Red)":   [rgb_per_kelas[l]["R"] for l in kelas_label],
        "G (Green)": [rgb_per_kelas[l]["G"] for l in kelas_label],
        "B (Blue)":  [rgb_per_kelas[l]["B"] for l in kelas_label],
    })
    fig_rgb = go.Figure()
    fig_rgb.add_trace(go.Bar(name="R", x=df_rgb["Kelas"], y=df_rgb["R (Red)"],    marker_color="#ef5350"))
    fig_rgb.add_trace(go.Bar(name="G", x=df_rgb["Kelas"], y=df_rgb["G (Green)"],  marker_color="#66bb6a"))
    fig_rgb.add_trace(go.Bar(name="B", x=df_rgb["Kelas"], y=df_rgb["B (Blue)"],   marker_color="#42a5f5"))
    fig_rgb.update_layout(
        barmode="group",
        title="Rata-rata Nilai RGB per Kelas",
        yaxis_title="Nilai Piksel (0–255)",
        legend_title="Channel",
    )
    st.plotly_chart(fig_rgb, use_container_width=True, key="bar_rgb")

    st.success("Nilai G (Green) yang tinggi pada kelas Healthy mengindikasikan daun yang lebih hijau dan segar dibanding kelas penyakit.")

# ─────────────────────────────────────────────
# HALAMAN 3: GALERI SAMPEL GAMBAR
# ─────────────────────────────────────────────
elif halaman == "🖼️ Image Samples":
    st.subheader("🖼️ Sampel Gambar Per Kelas")

    col_sel, col_jml = st.columns([2, 1])
    with col_sel:
        kelas_dipilih = st.selectbox("Pilih Kelas Penyakit:", kelas_label)
    with col_jml:
        pilihan_jumlah = st.selectbox(
            "Jumlah gambar:",
            options=list(range(1, 11)),
            index=3,   # default 4
            format_func=lambda x: f"{x} gambar"
        )
 
    folder_dipilih = label_to_folder[kelas_dipilih]
    folder_path    = os.path.join(BASE_PATH, "train", folder_dipilih)

    # Caption label kelas 
    warna = WARNA_KELAS[kelas_dipilih]
    st.markdown(f"""
        <div style="
            background-color: {warna}22;
            border-left: 5px solid {warna};
            padding: 12px 20px;
            border-radius: 8px;
            margin-bottom: 10px;
        ">
            <span style="font-size: 1.3rem; font-weight: bold; color: {warna};">
                🌿 Kelas: {kelas_dipilih}
            </span><br>
            <span style="color: #555; font-size: 0.9rem;">
                Menampilkan {pilihan_jumlah} sampel gambar dari data train
            </span>
        </div>
    """, unsafe_allow_html=True)
 
    if os.path.exists(folder_path):
        semua_gambar = (
            glob.glob(os.path.join(folder_path, "*.jpg")) +
            glob.glob(os.path.join(folder_path, "*.JPG"))
        )
        if semua_gambar:
            import random
            sampel = random.sample(semua_gambar, min(pilihan_jumlah, len(semua_gambar)))
            # Tampilkan maksimal 4 kolom per baris
            n_cols = min(pilihan_jumlah, 4)
            cols   = st.columns(n_cols)
            for i, path_gambar in enumerate(sampel):
                with cols[i % n_cols]:
                    img = Image.open(path_gambar)
                    st.image(img, caption=os.path.basename(path_gambar), use_container_width=True)
        else:
            st.warning("Tidak ada gambar ditemukan di folder ini.")
    else:
        st.error(f"Folder tidak ditemukan: `{folder_path}`")
 
# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.caption("🌾 RiceCare AI EDA Dashboard — Tim CC26-PSU169 | Coding Camp 2026 powered by DBS Foundation")
