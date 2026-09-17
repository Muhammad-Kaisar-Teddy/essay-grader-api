from typing import List, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nlp_pipeline.preprocessing import preprocess_text, NEGATION_WORDS

# ---------------------------------------------------------------------------
# Kata negasi aktif: jika ditemukan di jawaban siswa tapi TIDAK di referensi,
# atau sebaliknya, skor akan dipotong sesuai bobot.
# ---------------------------------------------------------------------------
_NEGATION_PENALTY = 0.35  # Penalti 35% per kata negasi yang tidak simetris


def calculate_lsa_score(student_answer: str, reference_answers: List[str]) -> float:
    """
    Menghitung skor LSA terhadap semua kunci jawaban yang diberikan guru.
    Mendukung multiple reference answers (misal: versi Bahasa Indonesia + Bahasa Inggris).

    Strategi:
    - Hitung cosine similarity antara jawaban siswa dengan SETIAP kunci jawaban.
    - Terapkan negation penalty jika ada asimetri penggunaan kata negasi.
    - Ambil skor TERTINGGI sebagai skor final LSA.
    """
    processed_student = preprocess_text(student_answer)
    if not processed_student:
        return 0.0

    best_score = 0.0
    for ref in reference_answers:
        processed_ref = preprocess_text(ref)
        if not processed_ref:
            continue
        score = _compute_similarity(processed_student, processed_ref)
        if score > best_score:
            best_score = score

    return best_score


def _extract_negations(text: str) -> Set[str]:
    """Ekstrak kata-kata negasi yang ada dalam teks."""
    return {word for word in text.split() if word in NEGATION_WORDS}


def _negation_penalty_factor(processed_student: str, processed_ref: str) -> float:
    """
    Hitung faktor penalti berdasarkan asimetri kata negasi.

    Logika:
    - Jika siswa menulis "tidak naik" tapi referensi "naik" → kata negasi ada di student,
      tidak ada di referensi → penalti diterapkan.
    - Jika simetris (keduanya pakai negasi atau keduanya tidak) → tidak ada penalti.

    Mengembalikan nilai antara 0.0 dan 1.0 (1.0 = tidak ada penalti).
    """
    student_negs = _extract_negations(processed_student)
    ref_negs = _extract_negations(processed_ref)

    # Kata negasi yang asimetris (muncul di salah satu sisi saja)
    asymmetric = student_negs.symmetric_difference(ref_negs)

    if not asymmetric:
        return 1.0  # Tidak ada penalti

    penalty = min(1.0, len(asymmetric) * _NEGATION_PENALTY)
    return 1.0 - penalty


def _compute_similarity(processed_student: str, processed_ref: str) -> float:
    """
    Menghitung kemiripan semantik antara dua teks yang sudah di-preprocess.

    Menggunakan:
    - TF-IDF dengan N-gram (1,2): bigram menangkap konteks negasi
      (contoh: "tidak naik" menjadi token tersendiri, berbeda dari "naik")
    - Cosine Similarity di ruang TF-IDF (lebih stabil dari SVD untuk 2 dokumen pendek)
    - Negation Penalty: aturan eksplisit untuk memotong skor jika ada asimetri negasi
    """
    # 1. Hitung negation penalty terlebih dahulu (sebelum vektorisasi)
    neg_factor = _negation_penalty_factor(processed_student, processed_ref)

    documents = [processed_ref, processed_student]

    # 2. TF-IDF dengan N-gram (1,2) untuk menangkap bigram negasi
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
    except ValueError:
        return 0.0

    n_features = tfidf_matrix.shape[1]
    if n_features == 0:
        return 0.0

    # 3. Cosine Similarity langsung di ruang TF-IDF N-gram
    # (lebih andal dari SVD saat hanya ada 2 dokumen pendek)
    sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    sim = float(max(0.0, min(1.0, sim)))

    # 4. Terapkan negation penalty
    final_score = sim * neg_factor * 100
    return final_score
