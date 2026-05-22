"""
Module: preprocessing
Description: Menangani pre-processing dokumen bahasa Indonesia menggunakan library Sastrawi.
"""
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Inisialisasi komponen Sastrawi di level modul agar hanya diload satu kali (menghemat memori & mempercepat proses)
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

stopword_factory = StopWordRemoverFactory()
stopwords_set = set(stopword_factory.get_stop_words())

def clean_text(text: str) -> str:
    """
    Menjalankan alur pre-processing teks secara berurutan:
    1. Case folding (mengubah ke huruf kecil).
    2. Punctuation removal (menghapus tanda baca dan angka).
    3. Stop-word removal (menghapus kata tidak bermakna).
    4. Stemming (mengubah kata menjadi kata dasar).
    
    Args:
        text (str): Teks mentah (dokumen atau kueri).
        
    Returns:
        str: Teks bersih hasil pre-processing yang digabungkan kembali menjadi string.
    """
    # Pastikan input adalah string
    if not isinstance(text, str):
        text = str(text)
        
    # 1. Case folding
    text = text.lower()
    
    # 2. Punctuation & Numbers removal (menyisakan hanya huruf a-z dan spasi)
    text = re.sub(r'\d+', ' ', text)           # Hapus angka
    text = re.sub(r'[^a-z\s]', ' ', text)      # Hapus tanda baca/karakter non-alfabet
    text = re.sub(r'\s+', ' ', text).strip()   # Normalisasi spasi berlebih
    
    # Memecah teks menjadi token untuk proses filter dan stemming
    tokens = text.split()
    
    # 3. Stop-word removal (juga menghapus karakter tunggal)
    filtered_tokens = [word for word in tokens if word not in stopwords_set and len(word) > 1]
    
    # 4. Stemming
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    
    # Menggabungkan kembali token menjadi satu string utuh
    final_text = ' '.join(stemmed_tokens)
    
    return final_text
