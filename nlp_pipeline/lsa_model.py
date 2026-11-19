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

    if flags is None:
        use_synonyms = False
    elif isinstance(flags, dict):
        use_synonyms = bool(flags.get("use_synonyms", False))
    else:
        use_synonyms = bool(getattr(flags, "use_synonyms", False))

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
    use_synonyms: bool = False
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
    # Pada perbandingan pasangan 2 dokumen (1 kunci vs 1 jawaban siswa),
    # gunakan k = min(tfidf_matrix.shape) (2 dimensi jika fitur >= 2)
    # agar proyeksi SVD tidak runtuh ke skalar 1D (yang membuat cosine similarity selalu 1.0).
    k = max(1, min(tfidf_matrix.shape))
    
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

def calculate_batch_lsa_scores(
    student_answers: List[str],
    reference_answers: List[str],
    convert_numbers: bool = True,
    flags: Optional[Any] = None
) -> List[float]:
    """
    Menghitung skor LSA secara BATCH-AWARE (Opsi B).
    Seluruh jawaban siswa dan kunci jawaban digabung menjadi satu korpus,
    lalu SVD di-fit SEKALI untuk mengekstrak ruang laten yang sebenarnya.
    """
    if not student_answers or not reference_answers:
        return [0.0] * len(student_answers)

    # 1. Preprocessing
    processed_students = [
        preprocess_text(ans, convert_numbers=convert_numbers, flags=flags) 
        for ans in student_answers
    ]
    processed_refs = [
        preprocess_text(ref, convert_numbers=convert_numbers, flags=flags) 
        for ref in reference_answers
    ]

    # Filter dokumen yang kosong setelah preprocessing
    valid_student_indices = [i for i, s in enumerate(processed_students) if s]
    valid_students = [processed_students[i] for i in valid_student_indices]
    valid_refs = [r for r in processed_refs if r]

    if not valid_students or not valid_refs:
        return [0.0] * len(student_answers)

    if flags is None:
        use_synonyms = False
    elif isinstance(flags, dict):
        use_synonyms = bool(flags.get("use_synonyms", False))
    else:
        use_synonyms = bool(getattr(flags, "use_synonyms", False))

    # 2. Global Synonym Alignment (Siswa ke Referensi)
    if use_synonyms:
        combined_ref = " ".join(valid_refs)
        aligned_students = [
            align_tokens_for_lsa(s, combined_ref)
            for s in valid_students
        ]
    else:
        aligned_students = valid_students

    # 3. Bangun Korpus Global (Referensi + Siswa)
    corpus = valid_refs + aligned_students
    num_refs = len(valid_refs)

    # 4. TF-IDF & SVD (Batch-Aware) pada Korpus
    num_unique_docs = len(set(corpus))
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))

    try:
        if num_unique_docs <= 2:
            # Jika seluruh jawaban siswa pada batch identik (atau hanya 1 siswa),
            # fit TF-IDF dan SVD 2D pada dokumen unik agar frekuensi kata tidak timpang
            # dan ruang laten tidak runtuh ke 1 dimensi.
            unique_corpus = list(dict.fromkeys(corpus))
            unique_tfidf = vectorizer.fit_transform(unique_corpus)
            n_features = unique_tfidf.shape[1]
            if n_features == 0:
                return [0.0] * len(student_answers)
            tfidf_matrix = vectorizer.transform(corpus)
            k = max(1, min(unique_tfidf.shape))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                svd = TruncatedSVD(n_components=k, random_state=42)
                svd.fit(unique_tfidf)
                latent_matrix = svd.transform(tfidf_matrix)
        else:
            tfidf_matrix = vectorizer.fit_transform(corpus)
            n_features = tfidf_matrix.shape[1]
            if n_features == 0:
                return [0.0] * len(student_answers)
            max_k = min(num_unique_docs - 1, n_features - 1)
            k = max(2, min(max_k, 50)) if max_k >= 2 else 1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                svd = TruncatedSVD(n_components=k, random_state=42)
                latent_matrix = svd.fit_transform(tfidf_matrix)

        # 6. Pisahkan matriks laten referensi dan siswa
        ref_latent = latent_matrix[:num_refs]
        student_latent = latent_matrix[num_refs:]

        # 7. Cosine Similarity (Siswa x Referensi)
        sim_matrix = cosine_similarity(student_latent, ref_latent)
    except Exception:
        return [0.0] * len(student_answers)

    # 8. Hitung Skor Akhir per Siswa (pilih skor tertinggi terhadap semua kunci jawaban)
    final_scores = [0.0] * len(student_answers)

    for idx_in_valid, original_idx in enumerate(valid_student_indices):
        best_score = 0.0
        for r_idx in range(num_refs):
            sim = float(max(0.0, min(1.0, float(sim_matrix[idx_in_valid][r_idx]))))
            neg_factor = _negation_penalty_factor(valid_students[idx_in_valid], valid_refs[r_idx])
            candidate_score = sim * neg_factor * 100.0
            if candidate_score > best_score:
                best_score = candidate_score

        final_scores[original_idx] = best_score

    return final_scores