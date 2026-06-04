import streamlit as st
import tensorflow as tf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
import re

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="Prediksi Judul Skripsi",
    page_icon="🎓",
    layout="wide"
)

# =====================================================
# CSS
# =====================================================

st.markdown("""
<style>

.stApp{
    background-color:#0F172A;
}

section[data-testid="stSidebar"]{
    background-color:#111827;
}

div[data-testid="metric-container"]{
    background:#1E293B;
    border-radius:15px;
    padding:15px;
}

h1,h2,h3{
    color:white;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
# =====================================================

from tensorflow.keras.models import load_model

model = load_model("model_ann.h5")

with open("tfidf.pkl","rb") as f:
    tfidf = pickle.load(f)

# =====================================================
# PREPROCESSING
# =====================================================

factory = StemmerFactory()
stemmer = factory.create_stemmer()

factory2 = StopWordRemoverFactory()
stopword = factory2.create_stop_word_remover()

def preprocessing(text):

    text = str(text)

    text = re.sub(r'[^a-zA-Z ]',' ',text)

    text = text.lower()

    text = stopword.remove(text)

    text = stemmer.stem(text)

    return text

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🎓 Dashboard")

st.sidebar.markdown("---")

st.sidebar.info("""
Artificial Neural Network (ANN)
""")

st.sidebar.success("""
TF-IDF Feature Extraction
""")

# =====================================================
# HEADER
# =====================================================

st.title("🎓 Dashboard Prediksi Judul Skripsi")

st.write(
"""
Sistem prediksi kategori judul skripsi menggunakan
Artificial Neural Network (ANN) dan TF-IDF.
"""
)

# =====================================================
# KPI
# =====================================================

k1,k2,k3,k4 = st.columns(4)

with k1:
    st.metric("Model","ANN")

with k2:
    st.metric("Metode","TF-IDF")

with k3:
    st.metric("Kelas","2")

with k4:
    st.metric("Status","Aktif")

st.markdown("---")

# =====================================================
# INPUT DAN HASIL
# =====================================================

left,right = st.columns([1,1])

with left:

    st.subheader("📝 Input Judul Skripsi")

    judul = st.text_area(
        "Masukkan Judul",
        height=200,
        placeholder="Contoh: Analisis Sentimen Twitter Menggunakan CNN"
    )

    prediksi_btn = st.button(
        "🚀 Prediksi"
    )

# =====================================================
# PREDIKSI
# =====================================================

if prediksi_btn:

    if judul.strip() == "":
        st.warning("Masukkan judul terlebih dahulu")

    else:

        hasil_preprocessing = preprocessing(judul)

        tfidf_input = tfidf.transform(
            [hasil_preprocessing]
        )

        prediksi = model.predict(
            tfidf_input.toarray(),
            verbose=0
        )

        prob_nonstem = float(prediksi[0][0])
        prob_stem = 1 - prob_nonstem

        if prob_nonstem >= 0.5:
            kategori = "NON STEM"
            confidence = prob_nonstem
        else:
            kategori = "STEM"
            confidence = prob_stem

        # ==========================================
        # HASIL KANAN
        # ==========================================

        with right:

            st.subheader("🎯 Hasil Prediksi")

            if kategori == "STEM":

                st.success(
                    f"Kategori : {kategori}"
                )

            else:

                st.warning(
                    f"Kategori : {kategori}"
                )

            st.subheader("📝 Hasil Preprocessing")

            st.info(
                hasil_preprocessing
            )

        st.markdown("---")

        # ==========================================
        # METRIC
        # ==========================================

        c1,c2 = st.columns(2)

        with c1:
            st.metric(
                "STEM",
                f"{prob_stem*100:.2f}%"
            )

        with c2:
            st.metric(
                "NON STEM",
                f"{prob_nonstem*100:.2f}%"
            )

        # ==========================================
        # PIE & GAUGE
        # ==========================================

        chart1,chart2 = st.columns(2)

        with chart1:

            st.subheader("🥧 Probabilitas")

            fig_pie = px.pie(
                names=[
                    "STEM",
                    "NON STEM"
                ],
                values=[
                    prob_stem*100,
                    prob_nonstem*100
                ],
                hole=0.6
            )

            st.plotly_chart(
                fig_pie,
                use_container_width=True
            )

        with chart2:

            st.subheader("⚡ Confidence")

            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=confidence*100,
                    title={
                        "text":"Confidence"
                    },
                    gauge={
                        "axis":{
                            "range":[0,100]
                        }
                    }
                )
            )

            st.plotly_chart(
                fig_gauge,
                use_container_width=True
            )

        # ==========================================
        # BAR CHART
        # ==========================================

        st.subheader(
            "📈 Perbandingan Probabilitas"
        )

        df_chart = pd.DataFrame({

            "Kategori":[
                "STEM",
                "NON STEM"
            ],

            "Persentase":[
                prob_stem*100,
                prob_nonstem*100
            ]

        })

        fig_bar = px.bar(
            df_chart,
            x="Persentase",
            y="Kategori",
            orientation="h",
            text="Persentase"
        )

        st.plotly_chart(
            fig_bar,
            use_container_width=True
        )

        # ==========================================
        # PROGRESS
        # ==========================================

        st.subheader(
            "📊 Tingkat Keyakinan Model"
        )

        st.progress(
            int(confidence*100)
        )

        st.write(
            f"Confidence : {confidence*100:.2f}%"
        )

        # ==========================================
        # KESIMPULAN
        # ==========================================

        st.subheader("📋 Ringkasan")

        st.info(
            f"""
            Model ANN memprediksi bahwa judul skripsi
            termasuk kategori {kategori}
            dengan tingkat keyakinan
            sebesar {confidence*100:.2f}%.
            """
        )