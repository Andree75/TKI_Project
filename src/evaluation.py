"""
Module: evaluation
Description: Menghitung metrik performa Information Retrieval (Precision, Recall, F-Measure).
"""

def calculate_precision(retrieved_docs: list[int], ground_truth_docs: list[int]) -> float:
    """
    Menghitung Precision.
    Precision = (Jumlah dokumen relevan yang terambil) / (Total dokumen yang terambil)
    """
    if not retrieved_docs:
        return 0.0
    
    # Mencari irisan (dokumen yang terambil DAN memang relevan/ada di ground truth)
    relevant_retrieved = set(retrieved_docs).intersection(set(ground_truth_docs))
    
    precision = len(relevant_retrieved) / len(retrieved_docs)
    return precision


def calculate_recall(retrieved_docs: list[int], ground_truth_docs: list[int]) -> float:
    """
    Menghitung Recall.
    Recall = (Jumlah dokumen relevan yang terambil) / (Total semua dokumen relevan)
    """
    if not ground_truth_docs:
        return 0.0
        
    # Mencari irisan
    relevant_retrieved = set(retrieved_docs).intersection(set(ground_truth_docs))
    
    recall = len(relevant_retrieved) / len(ground_truth_docs)
    return recall


def calculate_f_measure(precision: float, recall: float) -> float:
    """
    Menghitung F-Measure (F1-Score).
    Ini adalah rata-rata harmonik dari Precision dan Recall.
    Rumus: 2 * (Precision * Recall) / (Precision + Recall)
    """
    # Menangani error pembagian dengan nol (Division by Zero)
    if (precision + recall) == 0:
        return 0.0
        
    f_measure = 2 * (precision * recall) / (precision + recall)
    return f_measure


def calculate_all_metrics(retrieved_docs: list[int], ground_truth_docs: list[int]) -> dict:
    """
    Fungsi pembantu (helper) untuk menghitung ketiga metrik sekaligus dalam format persentase (%).
    """
    p = calculate_precision(retrieved_docs, ground_truth_docs)
    r = calculate_recall(retrieved_docs, ground_truth_docs)
    f1 = calculate_f_measure(p, r)
    
    return {
        'precision': p * 100,
        'recall': r * 100,
        'f_measure': f1 * 100
    }
