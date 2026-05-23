import streamlit as st
import pandas as pd
import numpy as np
import math

# Import modul-modul modular yang sudah kita buat
from src.preprocessing import clean_text
from src.engine import (
    build_inverted_index, calculate_idf, calculate_tfidf_matrix,
    vectorize_query, compute_cosine_similarity, compute_dot_product
)
from src.evaluation import calculate_all_metrics

# ====================================================================
# 1. KONFIGURASI HALAMAN & ESTETIKA (PREMIUM UI)
# ====================================================================
st.set_page_config(page_title="IR Mini Search Engine", page_icon="🔍", layout="wide")

st.markdown("""
<style>
    /* Styling Premium Aesthetic */
    .title-box {
        padding: 30px; 
        border: 2px solid #4CAF50; 
        border-radius: 15px; 
        background: linear-gradient(135deg, #ffffff, #e8f5e9); 
        text-align: center; 
        font-family: 'Inter', 'Segoe UI', Tahoma, sans-serif;
        box-shadow: 0 4px 15px rgba(46,125,50,0.15);
        margin-bottom: 30px;
    }
    .title-box h1 {
        color: #2E7D32; 
        margin-bottom: 10px;
        font-weight: 800;
        font-size: 2.5rem;
    }
    .title-box p {
        font-size: 1.2rem;
        color: #555;
    }
    .doc-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #4CAF50;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .doc-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(46,125,50,0.15);
    }
    .score-badge {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 6px 15px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9em;
        border: 1px solid #4CAF50;
    }
    .report-card {
        background-color: #FAFAFA;
        padding: 25px;
        border-radius: 10px;
        border: 1px solid #E0E0E0;
        margin-bottom: 20px;
    }
</style>
<div class="title-box">
    <h1>Mesin Pencari TKI - Ketahanan Pangan 🌾</h1>
    <p>Dikembangkan oleh <b>Andri Darmawan</b> (301210004) & <b>Muhammad Fakhrudin</b> (3012310043)</p>
</div>
""", unsafe_allow_html=True)


# ====================================================================
# 2. DATA LOADING & PIPELINE ENGINE (CACHED)
# ====================================================================
@st.cache_data
def load_data():
    """Memuat data langsung dari file lokal hasil_vsm_ketahanan_pangan.xlsx"""
    try:
        df = pd.read_excel('hasil_vsm_ketahanan_pangan.xlsx')
        if len(df.columns) >= 2:
            df.columns = ['Komentar', 'Sumber'] + list(df.columns[2:])
        else:
            df.columns = ['Komentar']
        df = df.dropna(subset=['Komentar']).reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Gagal memuat file Excel: {e}")
        return pd.DataFrame({'Komentar': []})

@st.cache_resource
def prepare_search_engine(documents):
    """Membangun index, IDF, dan Matriks TF-IDF secara efisien"""
    if not documents:
        return {}, {}, {}, np.array([]), []
        
    total_docs = len(documents)
    # 1. Bangun Inverted Index
    inverted_index, df_dict, processed_docs = build_inverted_index(documents)
    # 2. Hitung IDF Weights
    idf_weights = calculate_idf(df_dict, total_docs)
    # 3. Hitung Matriks TF-IDF
    doc_matrix, vocab = calculate_tfidf_matrix(inverted_index, idf_weights, total_docs)
    
    return inverted_index, df_dict, idf_weights, doc_matrix, vocab

# --- Menjalankan Pipeline ---
df = load_data()
documents = df['Komentar'].astype(str).tolist()

with st.spinner('Menyiapkan Engine Pencarian (Pre-processing, TF-IDF, Inverted Index)...'):
    inverted_index, df_dict, idf_weights, doc_matrix, vocab = prepare_search_engine(documents)


# ====================================================================
# 3. ANTARMUKA PENGGUNA (2 TAB UTAMA)
# ====================================================================
tab1, tab2 = st.tabs(["🔍 Mesin Pencari", " Laporan Analisis"])

# --------------------------------------------------------------------
# TAB 1: 🔍 MESIN PENCARI
# --------------------------------------------------------------------
with tab1:
    st.markdown("### 🔎 Cari Dokumen Relevan")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        query = st.text_input("Masukkan Kueri Pencarian:", placeholder="Contoh: harga beras stabil murah")
    
    with col2:
        top_k = st.slider("Jumlah Hasil (Top-K):", min_value=1, max_value=15, value=5)
        # Checkbox opsional untuk Analisis 2b
        use_dot_product = st.checkbox("Gunakan Dot Product (Tanpa Normalisasi)", 
                                      help="Menghilangkan Cosine Normalization untuk melihat efek Length Bias. (Lihat analisis lengkap di Tab Laporan)")

    if query:
        with st.spinner('Menghitung Skor Kemiripan...'):
            q_vector = vectorize_query(query, vocab, idf_weights)
            
            if use_dot_product:
                scores = compute_dot_product(doc_matrix, q_vector)
                score_label = "Dot Product Score"
            else:
                scores = compute_cosine_similarity(doc_matrix, q_vector)
                score_label = "Cosine Similarity"
            
            ranked_indices = scores.argsort()[::-1][:top_k]
            chart_data = {"Dokumen": [], "Skor": []}

            st.markdown("---")
            st.markdown("### Hasil Pencarian Peringkat Teratas")
            
            found = False
            for rank, idx in enumerate(ranked_indices, start=1):
                skor_dokumen = scores[idx]
                if skor_dokumen > 0:
                    found = True
                    teks_asli = df['Komentar'].iloc[idx]
                    sumber = df.get('Sumber', pd.Series(['-'] * len(df))).iloc[idx]
                    
                    chart_data["Dokumen"].append(f"Doc {idx}")
                    chart_data["Skor"].append(skor_dokumen)
                    
                    st.markdown(f"""
                        <div class="doc-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                <h4 style="margin: 0; color: #2E7D32;">Peringkat #{rank} | Dokumen ID: {idx}</h4>
                                <span class="score-badge">{score_label}: {skor_dokumen:.4f}</span>
                            </div>
                            <p style="font-size: 1.1em; line-height: 1.5; color: #444;">"{teks_asli}"</p>
                            <small style="color: #888;"><b>Sumber:</b> {sumber}</small>
                        </div>
                    """, unsafe_allow_html=True)
            
            if not found:
                st.warning("Tidak ditemukan dokumen yang relevan dengan kueri Anda.")
            else:
                st.markdown("### Perbandingan Skor Dokumen (Visualisasi)")
                df_chart = pd.DataFrame(chart_data).set_index("Dokumen")
                st.bar_chart(df_chart, use_container_width=True, color="#4CAF50")


# --------------------------------------------------------------------
# TAB 2: LAPORAN ANALISIS TEORI (SOAL UTS)
# --------------------------------------------------------------------
with tab2:
    st.header("Laporan Analisis & Teori UTS")
    st.markdown("Halaman ini menyajikan analisis teori interaktif untuk menjawab soal laporan evaluasi UTS.")
    
    # --- BAGIAN 1: Kalkulator IDF Interaktif (Soal 2a) ---
    st.markdown("""<div class="report-card">""", unsafe_allow_html=True)
    st.subheader("Bagian 1: Kalkulator IDF Interaktif (Soal 2a)")
    st.write("Masukkan dua kata kunci untuk menghitung nilai *Inverse Document Frequency* (IDF) secara dinamis dan membandingkan bobotnya.")
    
    col_input1, col_input2 = st.columns(2)
    with col_input1:
        kata_1_raw = st.text_input("Kata Kunci Pertama [A]:", value="beras")
    with col_input2:
        kata_2_raw = st.text_input("Kata Kunci Kedua [B]:", value="stabil")
        
    # Pre-processing kata agar seragam dengan vocabulary Inverted Index (stemming Sastrawi & lowercasing)
    kata_1_clean = clean_text(kata_1_raw).strip()
    kata_2_clean = clean_text(kata_2_raw).strip()
    
    # Ambil kata pertama saja jika user memasukkan lebih dari satu kata
    kata_1 = kata_1_clean.split()[0] if kata_1_clean else ""
    kata_2 = kata_2_clean.split()[0] if kata_2_clean else ""
    
    N_docs = 50 # Sesuai ketentuan di soal UTS
    
    # Ekstraksi Frekuensi Dokumen (df) dari memori sistem
    df_1 = df_dict.get(kata_1, 0) if kata_1 else 0
    df_2 = df_dict.get(kata_2, 0) if kata_2 else 0
    
    # Kalkulasi IDF
    idf_1 = math.log10(N_docs / df_1) if df_1 > 0 else 0.0
    idf_2 = math.log10(N_docs / df_2) if df_2 > 0 else 0.0
    
    # Fungsi pembantu untuk merender rumus LaTeX
    def render_idf_latex(kata, df, idf):
        if not kata:
            return
        if df == 0:
            st.warning(f"Kata tidak ditemukan dalam database dokumen ($df = 0$).")
            return
            
        ratio = N_docs / df
        st.latex(rf"\text{{IDF}}(\text{{{kata}}}) = \log_{{10}}\left(\frac{{{N_docs}}}{{{df}}}\right)")
        st.latex(rf"\text{{IDF}}(\text{{{kata}}}) = \log_{{10}}({ratio:.4f})")
        st.latex(rf"\text{{Hasil}} = {idf:.4f}")

    st.markdown("---")
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.markdown(f"**Perhitungan Kata [A]: '{kata_1_raw}'** *(Stem: {kata_1})*")
        st.write(f"- Total Dokumen ($N$): **{N_docs}**")
        st.write(f"- Frekuensi Dokumen ($df$): **{df_1}**")
        render_idf_latex(kata_1, df_1, idf_1)
        
    with col_res2:
        st.markdown(f"**Perhitungan Kata [B]: '{kata_2_raw}'** *(Stem: {kata_2})*")
        st.write(f"- Total Dokumen ($N$): **{N_docs}**")
        st.write(f"- Frekuensi Dokumen ($df$): **{df_2}**")
        render_idf_latex(kata_2, df_2, idf_2)
        
    # --- Kesimpulan Otomatis ---
    st.markdown("##### Kesimpulan Analisis Otomatis")
    if not kata_1 or not kata_2:
        st.info("Silakan masukkan kedua kata untuk melihat analisis perbandingan otomatis.")
    elif df_1 == 0 or df_2 == 0:
        st.info("Salah satu atau kedua kata tidak dikenali oleh sistem (df=0), sehingga perbandingan tidak dapat dilakukan.")
    elif idf_1 == idf_2:
        st.success(f"Kata **'{kata_1_raw}'** dan **'{kata_2_raw}'** memiliki nilai IDF yang sama persis karena keduanya muncul di jumlah dokumen yang sama ($df = {df_1}$). Keduanya memberikan bobot informasi yang seimbang.")
    else:
        # Menentukan kata mana yang lebih tinggi IDF-nya
        if idf_1 > idf_2:
            kata_tinggi, kata_rendah = kata_1_raw, kata_2_raw
        else:
            kata_tinggi, kata_rendah = kata_2_raw, kata_1_raw
            
        st.success(f"Kata **[{kata_tinggi}]** memiliki IDF lebih tinggi daripada **[{kata_rendah}]** karena muncul di lebih sedikit dokumen, yang membuktikan secara matematis bahwa semakin unik sebuah kata, semakin tinggi bobot informasinya dalam membedakan konteks dokumen.")
        
    st.markdown("""</div>""", unsafe_allow_html=True)
    
    
    # --- BAGIAN 2: Analisis Efek Normalisasi (Untuk soal 2b) ---
    st.markdown("""<div class="report-card">""", unsafe_allow_html=True)
    st.subheader("Bagian 2: Analisis Efek Normalisasi (Soal 2b)")
    
    st.markdown("""
    **Analisis Fenomena *Length Bias* (Dot Product vs Cosine Normalization)**
    
    Berdasarkan teori *Vector Space Model* (VSM), perhitungan kemiripan kueri terhadap dokumen dapat dilakukan melalui Perkalian Titik (*Dot Product*). Namun, perhitungan **Dot Product murni memiliki kelemahan yang disebut *Length Bias***.
    
    Dokumen teks yang panjang secara natural memiliki kuantitas kata yang lebih banyak, yang akan memperbesar nilai kemunculan kata (*tf*) di dalam dokumen tersebut. Akibatnya, dokumen panjang akan secara otomatis memperoleh skor kemiripan (dot product) yang jauh lebih tinggi secara kuantitas, padahal isi kontennya belum tentu lebih relevan secara kualitas dibanding dokumen yang lebih pendek.
    
    **Solusi: Cosine Normalization**
    Untuk menetralisir efek *Length Bias* ini, kita membagi hasil *Dot Product* dengan perkalian panjang (norma Euclidean) dari vektor Dokumen dan vektor Kueri:
    """)
    
    st.latex(r"\text{Cosine Similarity} = \frac{\vec{D} \cdot \vec{Q}}{||\vec{D}|| \times ||\vec{Q}||}")
    
    st.markdown("""
    Melalui pembagian norma ini, seluruh vektor dokumen diproyeksikan (dinormalisasi) ke dalam radius sudut yang sama (vektor unit = 1). Dampaknya, **dokumen pendek yang isinya padat, sangat spesifik, dan tepat sasaran** dapat memenangkan peringkat (ranking) dan bersaing secara adil melawan dokumen panjang yang mungkin banyak mengandung kata-kata tidak relevan.
    """)
    
    st.info("""
    **BUKTIKAN SECARA LANGSUNG!** 
    Silakan menuju **Tab 1: 🔍 Mesin Pencari**, masukkan kueri kalimat utuh, lalu aktifkan kotak centang (*checkbox*) **"Gunakan Dot Product (Tanpa Normalisasi)"**. Anda akan langsung melihat bagaimana susunan peringkat dokumen dan besaran nilai skornya (*Dot Product vs Cosine*) berubah secara drastis!
    """)
    st.markdown("""</div>""", unsafe_allow_html=True)
    
    
    # --- BAGIAN 3: Evaluasi Sistem Dinamis (Untuk soal 2c) ---
    st.markdown("""<div class="report-card">""", unsafe_allow_html=True)
    st.subheader("Bagian 3: Kalkulator Evaluasi Sistem (Soal 2c)")
    st.write("Mekanisme Evaluasi: Ketik Kueri terlebih dahulu untuk melihat dokumen yang dipanggil oleh sistem, kemudian tentukan Ground Truth berdasarkan hasil tersebut.")
    
    # Fungsi pembantu lokal untuk menjalankan pencarian kustom pada menu evaluasi
    def jalankan_retrieval_eval(q_text):
        if not q_text:
            return []
        q_vec = vectorize_query(q_text, vocab, idf_weights)
        scores = compute_cosine_similarity(doc_matrix, q_vec)
        # Mengembalikan list ID dokumen teratas dengan skor > 0 (Maksimal 5 dokumen)
        return [int(idx) for idx in scores.argsort()[::-1] if scores[idx] > 0][:5]

    # Membuat layout 2 kolom untuk Skenario 1 dan Skenario 2
    col_eval1, col_eval2 = st.columns(2)
    
    # --- KOLOM KIRI: SKENARIO 1 ---
    with col_eval1:
        st.markdown("#### Skenario Pengujian 1")
        query_1 = st.text_input("Masukkan Kueri 1:", value="harga beras murah", key="q1_eval")
        
        top_retrieved_1 = []
        if query_1:
            # 1. Menampilkan output dokumen sistem terlebih dahulu
            top_retrieved_1 = jalankan_retrieval_eval(query_1)
            st.success(f"**[Sistem] ID Terpanggil:** `{top_retrieved_1}`")
            
            st.write("**Pratinjau Dokumen Terpanggil:**")
            for idx in top_retrieved_1:
                st.markdown(f"- **[ID {idx}]** \"{df['Komentar'].iloc[idx][:80]}...\"")
            
            st.markdown("---")
            # 2. Input Ground Truth baru muncul/diisi setelah melihat output di atas
            gt_1 = st.text_input("Masukkan Ground Truth Kueri 1 (pisahkan dengan koma):", value="22, 41", key="gt1_eval")
        else:
            gt_1 = ""

    # --- KOLOM KANAN: SKENARIO 2 ---
    with col_eval2:
        st.markdown("#### Skenario Pengujian 2")
        query_2 = st.text_input("Masukkan Kueri 2:", value="harga beras tidak stabil", key="q2_eval")
        
        top_retrieved_2 = []
        if query_2:
            # 1. Menampilkan output dokumen sistem terlebih dahulu
            top_retrieved_2 = jalankan_retrieval_eval(query_2)
            st.success(f"**[Sistem] ID Terpanggil:** `{top_retrieved_2}`")
            
            st.write("**Pratinjau Dokumen Terpanggil:**")
            for idx in top_retrieved_2:
                st.markdown(f"- **[ID {idx}]** \"{df['Komentar'].iloc[idx][:80]}...\"")
            
            st.markdown("---")
            # 2. Input Ground Truth baru muncul/diisi setelah melihat output di atas
            gt_2 = st.text_input("Masukkan Ground Truth Kueri 2 (pisahkan dengan koma):", value="4, 19", key="gt2_eval")
        else:
            gt_2 = ""

    st.markdown("---")
    
    # Tombol kalkulasi akhir menggunakan matriks evaluasi
    if query_1 and query_2:
        if st.button("Hitung Evaluasi Matriks", use_container_width=True):
            eval_results = []
            
            # Memproses metrik skenario 1
            if gt_1:
                gt_ids_1 = [int(x.strip()) for x in gt_1.split(',') if x.strip().isdigit()]
                metrics_1 = calculate_all_metrics(top_retrieved_1, gt_ids_1)
                eval_results.append({
                    "Skenario": "Skenario 1",
                    "Kueri": query_1,
                    "Retrieved (Top 5)": str(top_retrieved_1),
                    "Ground Truth": str(gt_ids_1),
                    "Precision": f"{metrics_1['precision']:.1f}%",
                    "Recall": f"{metrics_1['recall']:.1f}%",
                    "F-Measure": f"{metrics_1['f_measure']:.1f}%"
                })
                
            # Memproses metrik skenario 2
            if gt_2:
                gt_ids_2 = [int(x.strip()) for x in gt_2.split(',') if x.strip().isdigit()]
                metrics_2 = calculate_all_metrics(top_retrieved_2, gt_ids_2)
                eval_results.append({
                    "Skenario": "Skenario 2",
                    "Kueri": query_2,
                    "Retrieved (Top 5)": str(top_retrieved_2),
                    "Ground Truth": str(gt_ids_2),
                    "Precision": f"{metrics_2['precision']:.1f}%",
                    "Recall": f"{metrics_2['recall']:.1f}%",
                    "F-Measure": f"{metrics_2['f_measure']:.1f}%"
                })
                
            if eval_results:
                st.markdown("#### Tabel Indikator Akurasi Hasil Akhir")
                st.dataframe(pd.DataFrame(eval_results), use_container_width=True)
                st.success("Tabel evaluasi performa berhasil diperbarui!")
    else:
        st.info("Silakan lengkapi Kueri Skenario 1 dan Kueri Skenario 2 terlebih dahulu.")

    st.markdown("""</div>""", unsafe_allow_html=True)