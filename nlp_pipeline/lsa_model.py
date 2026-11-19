import warnings
from typing import List, Set, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nlp_pipeline.preprocessing import preprocess_text, NEGATION_WORDS
from nlp_pipeline.synonym_expander import align_tokens_for_lsa
from sklearn.decomposition import TruncatedSVD

# ---------------------------------------------------------------------------
# Kata negasi aktif: jika ditemukan di jawaban siswa tapi TIDAK di referensi,
# atau sebaliknya, skor akan dipotong sesuai bobot.
# ---------------------------------------------------------------------------
_NEGATION_PENALTY = 0.35  # Penalti 35% per kata negasi yang tidak simetris


def calculate_lsa_score(
    student_answer: str,
    reference_answers: List[str],
    convert_numbers: bool = True,
    flags: Optional[Any] = None
) -> float:
    """
    Menghitung skor LSA terhadap semua kunci jawaban yang diberikan guru.
    Mendukung multiple reference answers (misal: versi Bahasa Indonesia + Bahasa Inggris).

    Strategi:
    - Hitung cosine similarity antara jawaban siswa dengan SETIAP kunci jawaban.
    - Terapkan negation penalty jika ada asimetri penggunaan kata negasi.
    - Ambil skor TERTINGGI sebagai skor final LSA.
    """
    processed_student = preprocess_text(student_answer, convert_numbers=convert_numbers, flags=flags)
    if not processed_student:
        return 0.0

    use_synonyms = getattr(flags, "use_synonyms", True) if flags is not None else True

    best_score = 0.0
    for ref in reference_answers:
        processed_ref = preprocess_text(ref, convert_numbers=convert_numbers, flags=flags)
        if not processed_ref:
            continue
        score = _compute_similarity(processed_student, processed_ref, use_synonyms=use_synonyms)
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


def _compute_similarity(
    processed_student: str,
    processed_ref: str,
    use_synonyms: bool = True
) -> float:
    # 1. Hitung negation penalty terlebih dahulu (sebelum vektorisasi)
    neg_factor = _negation_penalty_factor(processed_student, processed_ref)

    # 1b. Penyelarasan token kanonik berbasis synset
    aligned_student = (
        align_tokens_for_lsa(processed_student, processed_ref)
        if use_synonyms
        else processed_student
    )

    documents = [processed_ref, aligned_student]

    # 2. TF-IDF dengan N-gram (1,2) untuk menangkap bigram negasi
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
    except ValueError:
        return 0.0

    n_features = tfidf_matrix.shape[1]
    if n_features == 0:
        return 0.0

    # 3. LSA SEJATI: Reduksi Dimensi dengan SVD
    k = max(1, min(tfidf_matrix.shape) - 1)
    
    try:
        # PERBAIKAN: Inisialisasi SVD dan fit_transform hanya DIEKSEKUSI SEKALI di dalam blok warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            svd = TruncatedSVD(n_components=k, random_state=42)
            latent_matrix = svd.fit_transform(tfidf_matrix)
        
        # 4. Cosine Similarity di ruang laten
        sim = cosine_similarity(latent_matrix[0:1], latent_matrix[1:2])[0][0]
        sim = float(max(0.0, min(1.0, sim)))
    except Exception:
        return 0.0

    # 5. Terapkan negation penalty
    final_score = sim * neg_factor * 100
    return final_score
