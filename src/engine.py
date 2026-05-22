"""
Module: engine
Description: Mengimplementasikan pembuatan Inverted Index, pembobotan TF-IDF kustom, 
             dan Vector Space Model (Cosine Similarity) murni dengan Numpy dan Dictionary.
"""
import math
import numpy as np
from collections import Counter
from src.preprocessing import clean_text

def build_inverted_index(documents: list[str]) -> tuple[dict, dict, list]:
    """
    Membangun struktur data Inverted Index dan Document Frequency (DF) dari sekumpulan dokumen.
    Secara internal memanggil `clean_text` untuk pre-processing setiap dokumen.
    
    Args:
        documents (list[str]): Daftar teks dokumen asli (belum diproses).
        
    Returns:
        tuple berisi:
        - inverted_index (dict): Format {term: {doc_id: tf_mentah}}
        - document_frequency (dict): Format {term: jumlah_dokumen_yang_mengandung_term}
        - processed_docs (list[list[str]]): Daftar token per dokumen setelah cleaning.
    """
    inverted_index = {}
    document_frequency = {}
    processed_docs = []
    
    for doc_id, text in enumerate(documents):
        # 1. Bersihkan dan tokenisasi teks menggunakan fungsi dari preprocessing.py
        cleaned_text = clean_text(text)
        tokens = cleaned_text.split()
        processed_docs.append(tokens)
        
        # 2. Hitung frekuensi kemunculan kata (TF Mentah) dalam dokumen ini
        term_counts = Counter(tokens)
        
        # 3. Perbarui Inverted Index dan Document Frequency
        for term, count in term_counts.items():
            if term not in inverted_index:
                inverted_index[term] = {}
                document_frequency[term] = 0
                
            inverted_index[term][doc_id] = count
            document_frequency[term] += 1
            
    return inverted_index, document_frequency, processed_docs

def calculate_idf(document_frequency: dict, total_docs: int) -> dict:
    """
    Menghitung bobot IDF untuk setiap kata menggunakan rumus logaritma basis 10.
    Rumus: IDF(t) = log10(N / df(t))
    
    Args:
        document_frequency (dict): Kamus frekuensi dokumen (DF) untuk tiap kata.
        total_docs (int): Jumlah total dokumen (N).
        
    Returns:
        dict: Pemetaan antara kata dan nilai IDF-nya {term: nilai_idf}
    """
    idf_weights = {}
    for term, df in document_frequency.items():
        if df > 0:
            idf_weights[term] = math.log10(total_docs / df)
        else:
            idf_weights[term] = 0.0
    return idf_weights

def calculate_tfidf_matrix(inverted_index: dict, idf_weights: dict, total_docs: int) -> tuple[np.ndarray, list]:
    """
    Membangun matriks bobot TF-IDF seluruh dokumen dengan Numpy.
    Syarat Wajib: Menggunakan Log frequency weighting untuk TF (1 + log10(tf) jika tf > 0).
    
    Args:
        inverted_index (dict): Inverted index.
        idf_weights (dict): Kamus nilai IDF dari fungsi calculate_idf.
        total_docs (int): Jumlah dokumen (N).
        
    Returns:
        tuple berisi:
        - tfidf_matrix (np.ndarray): Matriks TF-IDF berukuran (N, jumlah_kosakata).
        - vocabulary (list): Daftar kata unik yang berurutan alfabetis (menjadi penanda kolom).
    """
    # 1. Mendapatkan seluruh kosakata dan mengurutkannya secara abjad
    vocabulary = sorted(list(inverted_index.keys()))
    vocab_size = len(vocabulary)
    
    # Membuat kamus pemetaan dari "kata" ke "indeks kolom matriks"
    term_indices = {term: idx for idx, term in enumerate(vocabulary)}
    
    # Matriks nol (baris: dokumen, kolom: kata)
    tfidf_matrix = np.zeros((total_docs, vocab_size))
    
    # 2. Mengisi sel-sel matriks
    for term, postings in inverted_index.items():
        term_idx = term_indices[term]
        idf_val = idf_weights.get(term, 0.0)
        
        for doc_id, tf_mentah in postings.items():
            # Penerapan Rumus Log Frequency Weighting
            if tf_mentah > 0:
                tf_log = 1 + math.log10(tf_mentah)
                # Nilai akhir dalam matriks = (1 + log10(tf)) * IDF
                tfidf_matrix[doc_id, term_idx] = tf_log * idf_val
                
    return tfidf_matrix, vocabulary

def vectorize_query(query_text: str, vocabulary: list, idf_weights: dict) -> np.ndarray:
    """
    Mengonversi teks kueri pengguna menjadi Vektor TF-IDF.
    Penyamaan bobot kueri juga menggunakan: (1 + log10(tf_kueri)) * IDF.
    
    Args:
        query_text (str): Kueri asli.
        vocabulary (list): Kosakata dokumen referensi.
        idf_weights (dict): Kamus IDF dokumen.
        
    Returns:
        np.ndarray: Vektor 1 Dimensi TF-IDF kueri.
    """
    vocab_size = len(vocabulary)
    term_indices = {term: idx for idx, term in enumerate(vocabulary)}
    
    query_vector = np.zeros(vocab_size)
    
    # Pre-processing kueri dengan fungsi yang sama
    cleaned_query = clean_text(query_text)
    query_tokens = cleaned_query.split()
    
    # Menghitung TF mentah dari kueri
    query_tf_mentah = Counter(query_tokens)
    
    # Mengisi vektor kueri
    for term, tf_mentah in query_tf_mentah.items():
        if term in term_indices:  # Hanya memproses kata yang ada di database (vocabulary)
            term_idx = term_indices[term]
            idf_val = idf_weights.get(term, 0.0)
            
            # Log Frequency Weighting untuk kueri
            tf_log = 1 + math.log10(tf_mentah)
            query_vector[term_idx] = tf_log * idf_val
            
    return query_vector

def compute_cosine_similarity(doc_matrix: np.ndarray, query_vector: np.ndarray) -> np.ndarray:
    """
    Menghitung skor Cosine Similarity antara matriks seluruh dokumen dengan vektor kueri tunggal.
    Rumus Cosine Normalization: Cosine(D, Q) = (D . Q) / (||D|| * ||Q||)
    
    Args:
        doc_matrix (np.ndarray): Matriks TF-IDF dokumen.
        query_vector (np.ndarray): Vektor TF-IDF kueri.
        
    Returns:
        np.ndarray: Array skor similarity 1D (satu skor untuk setiap dokumen).
    """
    # 1. Menghitung Perkalian Titik (Dot Product)
    dot_product = np.dot(doc_matrix, query_vector)
    
    # 2. Menghitung Panjang Vektor (Euclidean Norm)
    # axis=1 berarti menghitung panjang vektor per baris dokumen
    norm_docs = np.linalg.norm(doc_matrix, axis=1)
    norm_query = np.linalg.norm(query_vector)
    
    # Mencegah error pembagian dengan nol (Division by Zero)
    # Jika panjang dokumen 0, set jadi 1.0
    norm_docs[norm_docs == 0] = 1.0 
    
    if norm_query == 0:
        # Jika kueri kosong atau tidak dikenali, return skor 0 untuk semua dokumen
        return np.zeros(doc_matrix.shape[0])
        
    # 3. Pembagian (Cosine Normalization)
    cosine_scores = dot_product / (norm_docs * norm_query)
    
    return cosine_scores

def compute_dot_product(doc_matrix: np.ndarray, query_vector: np.ndarray) -> np.ndarray:
    """
    Menghitung Dot Product tanpa Cosine Normalization.
    Tujuan: Diperlukan untuk Bagian 2b (Analisis Efek Normalisasi: Length Bias).
    """
    return np.dot(doc_matrix, query_vector)
