from nlp_pipeline.rule_based import calculate_rule_based_score
from nlp_pipeline.lsa_model import calculate_lsa_score
from nlp_pipeline.math_evaluator import calculate_math_score
from schemas import EssayRequest, GraderResponse


def grade_essay(request: EssayRequest) -> GraderResponse:
    """
    Mengorkestrasi proses penilaian esai hibrida:
    1. Rule-Based Keyword Matching (dengan bobot dinamis per keyword)
    2. LSA Cosine Similarity (terhadap semua kunci jawaban, ambil terbaik)
    3. Evaluasi Matematis SymPy (mengecek ekuivalensi langkah hitungan)
    4. Fusion: skor akhir = (rule_weight * rule_score) + (lsa_weight * lsa_score) + (math_weight * math_score)
    """
    # 1. Rule-based dengan bobot dinamis
    rule_score, matched = calculate_rule_based_score(request.student_answer, request.keywords)
    
    # 2. Evaluasi Matematis (jika ada math_steps)
    math_score = 0.0
    if request.math_steps:
        math_score = calculate_math_score(request.student_answer, request.math_steps)

    # 3. Hitung Skor Faktual (Srb) gabungan dari Keyword dan Math
    factual_weight = request.rule_weight + request.math_weight
    if factual_weight > 0:
        factual_score = ((rule_score * request.rule_weight) + (math_score * request.math_weight)) / factual_weight
    else:
        factual_score = 0.0

    # 4. Hybrid Fusion dengan Tgate dan Penalti (LSA baru dipanggil jika lolos Tgate)
    lsa_score = 0.0
    if factual_score >= request.tgate or request.tgate == 0:
        # Lulus Tgate: LSA AKTIF (Hitung sekarang)
        lsa_score = calculate_lsa_score(request.student_answer, request.reference_answers)
        
        # Agregasi berbobot penuh (S_final = α × Srb + β × Slsa)
        final_score = (rule_score * request.rule_weight) + (lsa_score * request.lsa_weight) + (math_score * request.math_weight)
    else:
        # Gagal Tgate: Penalti = Srb × (Srb / Tgate). LSA TIDAK DIJALANKAN SAMA SEKALI.
        penalty_score = factual_score * (factual_score / request.tgate)
        final_score = penalty_score * factual_weight
    
    # 5. Normalisasi bobot jika total bobot tidak 1.0 (opsional, tapi disarankan)
    total_weights = request.rule_weight + request.lsa_weight + request.math_weight
    if total_weights > 0:
        final_score = final_score / total_weights
    
    # Hitung nilai proporsional terhadap bobot soal maksimal
    scaled_score = (final_score / 100.0) * request.max_score

    return GraderResponse(
        final_score=round(final_score, 2),
        scaled_score=round(scaled_score, 2),
        rule_based_score=round(rule_score, 2),
        lsa_score=round(lsa_score, 2),
        math_score=round(math_score, 2),
        matched_keywords=matched
    )
