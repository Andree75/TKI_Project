import streamlit as st
import pandas as pd
import numpy as np
import re
import math
from collections import Counter
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ====================================================================
# CONFIG & PAGE SETUP
# ====================================================================
st.set_page_config(page_title="IR Mini Search Engine", page_icon="🔍", layout="wide")

st.markdown("""
    <div style="padding: 20px; border: 2px solid #4CAF50; border-radius: 10px; background-color: #f9f9f9; text-align: center; font-family: 'Segoe UI', sans-serif;">
        <h2 style="color: #2E7D32; margin-bottom: 5px;">Tugas Temu Kembali Informasi (TKI) - GUI Streamlit</h2>
        <p style="margin: 0;"><b>Andri Darmawan</b> (301210004) & <b>Muhammad Fakhrudin</b> (3012310043)</p>
        <hr style="border: 1px solid #4CAF50; width: 30%; margin: 10px auto;">
    </div>
""", unsafe_allow_html=True)

# ====================================================================
# CORE FUNCTIONS & PREPROCESSING (CACHED FOR PERFORMANCE)
# ====================================================================
@st.cache_resource
def init_sastrawi():
    stemmer = StemmerFactory().create_stemmer()
    stopwords = set(StopWordRemoverFactory().get_stop_words())
    return stemmer, stopwords

stemmer, stopwords = init_sastrawi()

def cleaning(teks):
    teks = str(teks).lower()
    teks = re.sub(r'\d+', ' ', teks)
    teks = re.sub(r'[^a-z\s]', ' ', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    return teks

def tokenisasi(teks):
    return [t for t in str(teks).split() if len(t) > 1]

def hapus_stopword(tokens):
    return [k for k in tokens if k not in stopwords]

def stemming(tokens):
    return [stemmer.stem(k) for k in tokens]

@st.cache_data
def load_and_index_dataset():
    # Load dataset dari link GitHub kamu
    url_data = 'https://github.com/muhfakhrudin/Tugas_TKI/raw/refs/heads/main/data_ketahanan_pangan_clean.xlsx'
    df = pd.read_excel(url_data)
    df.columns = ['Komentar', 'Sumber']
    df = df.dropna(subset=['Komentar']).reset_index(drop=True)
    
    # Preprocessing
    df['hasil_cleaning'] = [cleaning(t) for t in df['Komentar']]
    df['hasil_token'] = [tokenisasi(t) for t in df['hasil_cleaning']]
    df['hasil_stopword'] = [hapus_stopword(t) for t in df['hasil_token']]
    df['hasil_stemming'] = [stemming(t) for t in df['hasil_stopword']]
    df['teks_bersih'] = df['hasil_stemming'].apply(lambda x: ' '.join(x) if x else 'kosong')
    
    # Membangun Inverted Index
    inverted_index = {}
    for doc_id, tokens in enumerate(df['hasil_stemming']):
        hitung_kata = Counter(tokens)
        for term, tf_mentah in hitung_kata.items():
            if term not in inverted_index:
                inverted_index[term] = {}
            inverted_index[term][doc_id] = tf_mentah
            
    # Parameter VSM
    kosakata_vsm = sorted(list(inverted_index.keys()))
    dimensi_kata = len(kosakata_vsm)
    indeks_kata = {kata: i for i, kata in enumerate(kosakata_vsm)}
    
    N_docs = len(df)
    global_idf = {kata: math.log10(N_docs / len(posting)) for kata, posting in inverted_index.items()}
    
    # Membangun Matriks TF-IDF Kustom
    matriks_tfidf_kustom = np.zeros((N_docs, dimensi_kata))
    for kata, posting in inverted_index.items():
        kata_idx = indeks_kata[kata]
        idf_w = global_idf[kata]
        for doc_id, tf_mentah in posting.items():
            tf_log = 1 + math.log10(tf_mentah) if tf_mentah > 0 else 0
            matriks_tfidf_kustom[doc_id, kata_idx] = tf_log * idf_w
            
    return df, inverted_index, matriks_tfidf_kustom, global_idf, indeks_kata, dimensi_kata, N_docs

df, inverted_index, matriks_tfidf_kustom, global_idf, indeks_kata, dimensi_kata, N_docs = load_and_index_dataset()

# Helper untuk hitung metrik evaluasi
def hitung_metrik(retrieved, ground_truth):
    retrieved_set = set(retrieved)
    gt_set = set(ground_truth)
    relevan_terpanggil = retrieved_set.intersection(gt_set)
    
    precision = len(relevan_terpanggil) / len(retrieved_set) if len(retrieved_set) > 0 else 0
    recall = len(relevan_terpanggil) / len(gt_set) if len(gt_set) > 0 else 0
    f_measure = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision * 100, recall * 100, f_measure * 100

# ====================================================================
# INTERFACE NAVIGATION USING TABS
# ====================================================================
tab1, tab2 = st.tabs(["🔍 Nomor 1: Mini Search Engine", "📊 Nomor 2: Tugas Analisis & Evaluasi"])

# --------------------------------------------------------------------
# TAB 1: MINI SEARCH ENGINE
# --------------------------------------------------------------------
with tab1:
    st.header("Pengembangan Mesin Pencari Mini")
    
    # Input Kueri
    query_input = st.text_input("Masukkan kueri pencarian Anda:", placeholder="Contoh: harga beras murah stabil")
    top_k = st.slider("Jumlah Dokumen Teratas (Top-K):", min_value=1, max_value=10, value=5)
    
    if query_input:
        # Proses kueri
        q_clean = cleaning(query_input)
        q_tokens = tokenisasi(q_clean)
        q_stop = hapus_stopword(q_tokens)
        q_stem = stemming(q_stop)
        
        st.write(f"**Hasil Preprocessing Kueri:** `{q_stem}`")
        
        # Vektor Kueri
        q_counts = Counter(q_stem)
        vektor_query = np.zeros(dimensi_kata)
        for kata, tf_mentah in q_counts.items():
            if kata in indeks_kata:
                vektor_query[indeks_kata[kata]] = (1 + math.log10(tf_mentah)) * global_idf[kata]
        
        # Kalkulasi Skor Jarak
        skor_tanpa_norm = np.dot(matriks_tfidf_kustom, vektor_query)
        norm_query = np.linalg.norm(vektor_query)
        norm_dokumen = np.linalg.norm(matriks_tfidf_kustom, axis=1)
        norm_dokumen[norm_dokumen == 0] = 1.0
        skor_cosine = skor_tanpa_norm / (norm_query * norm_dokumen) if norm_query > 0 else np.zeros(N_docs)
        
        # Pengurutan Ranking
        urutan_cosine = skor_cosine.argsort()[::-1][:top_k]
        urutan_tanpa_norm = skor_tanpa_norm.argsort()[::-1][:top_k]
        
        # Tampilan Komparatif Berdampingan (Menjawab kebutuhan visual analisis)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🟢 Dengan Cosine Normalization")
            for rank, idx in enumerate(urutan_cosine, 1):
                if skor_cosine[idx] > 0:
                    with st.expander(f"Rank #{rank} | ID: {idx} | Skor: {skor_cosine[idx]:.4f}"):
                        st.write(df['Komentar'].iloc[idx])
                        st.caption(f"Sumber: {df['Sumber'].iloc[idx]}")
                        
        with col2:
            st.subheader("🔴 Tanpa Normalisasi (Dot Product)")
            for rank, idx in enumerate(urutan_tanpa_norm, 1):
                if skor_tanpa_norm[idx] > 0:
                    with st.expander(f"Rank #{rank} | ID: {idx} | Skor: {skor_tanpa_norm[idx]:.4f}"):
                        st.write(df['Komentar'].iloc[idx])
                        st.caption(f"Sumber: {df['Sumber'].iloc[idx]}")

# --------------------------------------------------------------------
# TAB 2: TUGAS ANALISIS & LAPORAN
# --------------------------------------------------------------------
with tab2:
    st.header("Laporan Analisis & Evaluasi Sistem")
    
    # --- SUBBAGIAN 2A ---
    st.subheader("2a. Analisis Bobot (IDF)")
    col_a1, col_a2 = st.columns([1, 2])
    with col_a1:
        kata_pilihan = st.selectbox("Pilih kata untuk simulasi hitung IDF manual:", ["pangan", "upaya", "beras", "stabil"])
        df_kata = len(inverted_index.get(kata_pilihan, {}))
        idf_kata = math.log10(50 / df_kata) if df_kata > 0 else 0
        
        st.metric(label=f"Document Frequency (df) '{kata_pilihan}'", value=df_kata)
        st.latex(r"IDF = \log_{10}\left(\frac{50}{" + str(df_kata) + r"}\right) = " + f"{idf_kata:.4f}")
        
    with col_a2:
        st.markdown(r"""
        **Mengapa kata yang lebih jarang muncul memiliki bobot lebih tinggi?**
        
        Secara matematis, nilai $df$ berada sebagai penyebut dalam logaritma pembagian variabel total dokumen ($\log_{10}(N/df)$). 
        * Jika kata sangat umum (seperti *'pangan'*, $df=25$), pembagi bernilai besar sehingga nilai rasio mengecil mendekati angka 1 ($\log_{10}(2) \approx 0.3010$). Kata ini kehilangan daya pembeda informasi.
        * Jika kata sangat langka (seperti *'upaya'*, $df=1$), rasio pembagian menjadi sangat besar ($\log_{10}(50) \approx 1.6990$). VSM memberikan penghargaan tinggi pada istilah spesifik ini karena membawa nilai informasi (*information gain*) unik yang membantu membedakan satu dokumen dengan dokumen lainnya.
        """)
        
    st.markdown("---")
    
    # --- SUBBAGIAN 2B ---
    st.subheader("2b. Analisis Efek Normalisasi")
    st.markdown(r"""
    **Mengapa dokumen panjang memiliki skor lebih tinggi jika tidak dinormalisasi?**
    
    1. **Length Bias:** Dokumen panjang memiliki kuantitas kata yang melimpah secara natural, memperbesar frekuensi kemunculan term ($tf$) mentah di dalamnya.
    2. **Akumulasi Dot Product:** Tanpa pembagian dengan panjang vektor ($\|\vec{D}\|$), perhitungan skor hanya mengandalkan $\sum (w_{t,Q} \times w_{t,D})$. Akibatnya, dokumen panjang diuntungkan secara tidak adil karena nilai penjumlahan bobotnya yang terus membengkak secara kuantitas kata, bukan karena kualitas kepadatan topiknya.
    3. **Solusi Cosine Normalization:** Metode ini membagi perkalian tersebut dengan norma Euclidean sehingga seluruh vektor dokumen diproyeksikan pada radius ruang bernilai sama (= 1). Dokumen pendek yang padat isi informasi relevan akhirnya dapat bersaing secara objektif dengan dokumen panjang yang bertele-tele.
    """)
    
    st.markdown("---")
    
    # --- SUBBAGIAN 2C ---
    st.subheader("2c. Evaluasi Sistem Dinamis")
    # Fungsi pembantu internal untuk memproses pencarian per skenario kueri
    def proses_retrieval_eval(query_text):
        if not query_text:
            return []
        q_clean = cleaning(query_text)
        q_tokens = tokenisasi(q_clean)
        q_stop = hapus_stopword(q_tokens)
        q_stem = stemming(q_stop)
        
        q_counts = Counter(q_stem)
        vektor_query = np.zeros(dimensi_kata)
        for kata, tf_mentah in q_counts.items():
            if kata in indeks_kata:
                vektor_query[indeks_kata[kata]] = (1 + math.log10(tf_mentah)) * global_idf[kata]
                
        skor_tanpa_norm = np.dot(matriks_tfidf_kustom, vektor_query)
        norm_query = np.linalg.norm(vektor_query)
        norm_dokumen = np.linalg.norm(matriks_tfidf_kustom, axis=1)
        norm_dokumen[norm_dokumen == 0] = 1.0
        skor_cosine = skor_tanpa_norm / (norm_query * norm_dokumen) if norm_query > 0 else np.zeros(N_docs)
        return [int(idx) for idx in skor_cosine.argsort()[::-1] if skor_cosine[idx] > 0][:5]

    # Membuat layout 2 kolom berdampingan untuk Skenario 1 dan Skenario 2
    col_sken1, col_sken2 = st.columns(2)

    # --- COLUMN KIRI: SKENARIO PENGUJIAN 1 ---
    with col_sken1:
        st.markdown("#### Skenario Pengujian 1")
        eval_query_1 = st.text_input("Masukkan Kueri Pengujian 1:", value="harga beras murah", key="eq1")
        top_retrieved_1 = proses_retrieval_eval(eval_query_1)
        
        if eval_query_1:
            st.write(f"**ID Terpanggil oleh Sistem (Kueri 1):** `{top_retrieved_1}`")
            for idx in top_retrieved_1:
                st.text(f" [{idx}] {df['Komentar'].iloc[idx][:65]}...")
        
        gt_input_1 = st.text_input("Masukkan ID Ground Truth Kueri 1:", value="22, 41", key="gt1")

    # --- COLUMN KANAN: SKENARIO PENGUJIAN 2 ---
    with col_sken2:
        st.markdown("#### Skenario Pengujian 2")
        eval_query_2 = st.text_input("Masukkan Kueri Pengujian 2:", value="harga beras tidak stabil", key="eq2")
        top_retrieved_2 = proses_retrieval_eval(eval_query_2)
        
        if eval_query_2:
            st.write(f"**ID Terpanggil oleh Sistem (Kueri 2):** `{top_retrieved_2}`")
            for idx in top_retrieved_2:
                st.text(f" [{idx}] {df['Komentar'].iloc[idx][:65]}...")
        
        gt_input_2 = st.text_input("Masukkan ID Ground Truth Kueri 2:", value="4, 19", key="gt2")

    st.markdown("---")

    # --- PERHITUNGAN & REKAPITULASI TABEL OTOMATIS ---
    st.write("#### Tabel Indikator Performa Akhir (Real-time Update)")

    rekap_data = []

    # Hitung metrik Skenario 1 jika input tersedia
    if eval_query_1 and gt_input_1:
        gt_list_1 = [int(x.strip()) for x in gt_input_1.split(",") if x.strip().isdigit()]
        p1, r1, f1 = hitung_metrik(top_retrieved_1, gt_list_1)
        rekap_data.append({
            "No": 1, 
            "Kueri": eval_query_1, 
            "Retrieved IDs": str(top_retrieved_1),
            "Ground Truth": str(gt_list_1), 
            "Precision": f"{p1:.1f}%", 
            "Recall": f"{r1:.1f}%", 
            "F1-Score": f"{f1:.1f}%"
        })

    # Hitung metrik Skenario 2 jika input tersedia
    if eval_query_2 and gt_input_2:
        gt_list_2 = [int(x.strip()) for x in gt_input_2.split(",") if x.strip().isdigit()]
        p2, r2, f2 = hitung_metrik(top_retrieved_2, gt_list_2)
        rekap_data.append({
            "No": 2, 
            "Kueri": eval_query_2, 
            "Retrieved IDs": str(top_retrieved_2),
            "Ground Truth": str(gt_list_2), 
            "Precision": f"{p2:.1f}%", 
            "Recall": f"{r2:.1f}%", 
            "F1-Score": f"{f2:.1f}%"
        })

    # Tampilkan ke dalam dataframe Streamlit jika data sudah siap
    if rekap_data:
        df_rekap_final = pd.DataFrame(rekap_data)
        st.dataframe(df_rekap_final, use_container_width=True)
    else:
        st.info("Silakan lengkapi input kueri dan Ground Truth di atas untuk memunculkan tabel indikator akhir.")