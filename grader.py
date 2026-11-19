from collections import defaultdict
from typing import List, Dict, Optional, Literal, Tuple, Any
from nlp_pipeline.rule_based import calculate_rule_based_score
from nlp_pipeline.lsa_model import calculate_lsa_score, calculate_batch_lsa_scores
from nlp_pipeline.math_evaluator import calculate_math_score
from nlp_pipeline.preprocessing import clean_html_and_newlines
from schemas import (
    EssayRequest,
    GraderResponse,
    BatchEssayItem,
    BatchEssayRequest,
    BatchGraderResponse,
    BatchGraderResultItem,
    KeywordItem,
    MathStepItem,
    PreprocessingFlags,
)


def _compute_factual_score(
    rule_score: float,
    math_score: float,
    rule_weight: float,
    math_weight: float,
    has_keywords: bool,
    has_answer: bool,
) -> float:
    """
    Menghitung skor faktual gabungan dari Rule-Based dan Math Evaluation.
    Jika bobot faktual (rule_weight + math_weight) > 0, dihitung rata-rata tertimbangnya.
    Jika bobot faktual == 0 (mode murni LSA, mis. rule_weight=0, lsa_weight=1),
    gunakan rule_score jika ada kata kunci, atau 100.0 jika jawaban tidak kosong.
    """
    factual_weight = rule_weight + math_weight
    if factual_weight > 0:
        return ((rule_score * rule_weight) + (math_score * math_weight)) / factual_weight
    if has_keywords:
        return rule_score
    return 100.0 if has_answer else 0.0


def grade_essay(request: EssayRequest) -> GraderResponse:
    """
    Mengorkestrasi proses penilaian esai hibrida (tunggal):
    1. Text Preprocessing & Normalisasi HTML
    2. Rule-Based Keyword Matching (dengan Synonym Expansion & bobot dinamis)
    3. Evaluasi Matematis SymPy (jika ada math_steps)
    4. Pengecekan Threshold Gate (Tgate) pada Skor Faktual
    5. LSA Cosine Similarity (jika lolos Tgate)
    6. Normalisasi Skor Akhir terhadap max_score
    """
    flags = request.get_preprocessing_flags()

    # 0. Normalisasi HTML dan jeda baris
    if flags.clean_html:
        clean_student_answer = clean_html_and_newlines(request.student_answer)
        clean_reference_answers = [clean_html_and_newlines(ref) for ref in request.reference_answers]
        inner_flags = flags.model_copy(update={"clean_html": False})
    else:
        clean_student_answer = request.student_answer
        clean_reference_answers = request.reference_answers
        inner_flags = flags

    # 1. Rule-based dengan bobot dinamis, synonym expansion, dan feature flags
    rule_score, matched = calculate_rule_based_score(
        clean_student_answer,
        request.keywords,
        request.required_keywords_count,
        flags=inner_flags
    )

    # 2. Evaluasi Matematis (jika ada math_steps)
    math_score = 0.0
    if request.math_steps:
        math_score = calculate_math_score(
            clean_student_answer,
            request.math_steps,
            reference_answers=clean_reference_answers,
            keywords=request.keywords,
        )

    # 3. Hitung Skor Faktual (S_fac) gabungan dari Keyword dan Math
    factual_score = _compute_factual_score(
        rule_score=rule_score,
        math_score=math_score,
        rule_weight=request.rule_weight,
        math_weight=request.math_weight,
        has_keywords=bool(request.keywords),
        has_answer=bool(clean_student_answer and clean_student_answer.strip()),
    )

    # 4. Hybrid Fusion dengan Tgate
    lsa_score = 0.0
    if (factual_score > 0) and (request.tgate == 0 or factual_score >= request.tgate):
        # Lulus Tgate: LSA AKTIF
        lsa_score = calculate_lsa_score(
            clean_student_answer,
            clean_reference_answers,
            flags=inner_flags
        )
        final_score = (
            (rule_score * request.rule_weight)
            + (lsa_score * request.lsa_weight)
            + (math_score * request.math_weight)
        )
    else:
        # Gagal Tgate (atau skor faktual 0): LSA TIDAK DIJALANKAN -> return skor faktual
        final_score = (rule_score * request.rule_weight) + (math_score * request.math_weight)

    # 5. Normalisasi bobot & konversi ke skala max_score
    total_weights = request.rule_weight + request.lsa_weight + request.math_weight
    if total_weights > 0:
        final_score = final_score / total_weights

    scaled_score = (final_score / 100.0) * request.max_score

    return GraderResponse(
        final_score=round(final_score, 2),
        scaled_score=round(scaled_score, 2),
        rule_based_score=round(rule_score, 2),
        lsa_score=round(lsa_score, 2),
        math_score=round(math_score, 2),
        matched_keywords=matched,
        essay_mode=request.resolve_essay_mode()
    )


def grade_batch_by_question(
    student_answers: List[str],
    reference_answers: List[str],
    keywords: List[KeywordItem],
    math_steps: List[MathStepItem],
    rule_weight: float,
    lsa_weight: float,
    math_weight: float,
    tgate: float,
    max_score: float = 100.0,
    required_keywords_count: Optional[int] = None,
    subject: Optional[str] = None,
    essay_mode: Optional[Literal["linguistic", "exact", "conceptual"]] = None,
    flags: Optional[PreprocessingFlags] = None,
) -> List[GraderResponse]:
    """
    Menilai sekumpulan jawaban seluruh siswa UNTUK SATU SOAL YANG SAMA (satu id_soal)
    secara berurutan:
    1. Text Preprocessing & Normalisasi HTML
    2. Synonym Expansion & Rule-Based Keyword Matching (+ Math Evaluation jika ada)
    3. Pengecekan Tgate pada Skor Faktual:
       - Jika Tgate TIDAK terpenuhi -> langsung return skor faktual (LSA = 0)
       - Jika Tgate TERPENUHI -> masuk ke kandidat LSA Sejati
    4. LSA Sejati (Batch-Aware Corpus SVD):
       - Membangun matriks TF-IDF & TruncatedSVD dari korpus gabungan
         (Kunci Jawaban + Jawaban Siswa pada id_soal tersebut)
    5. Normalisasi skor akhir berdasarkan max_score soal.
    """
    if not student_answers:
        return []

    temp_req = EssayRequest(
        student_answer="",
        reference_answers=reference_answers,
        keywords=keywords,
        required_keywords_count=required_keywords_count,
        math_steps=math_steps,
        rule_weight=rule_weight,
        lsa_weight=lsa_weight,
        math_weight=math_weight,
        tgate=tgate,
        max_score=max_score,
        subject=subject,
        essay_mode=essay_mode,
        preprocessing_flags=flags,
    )
    actual_flags = temp_req.get_preprocessing_flags()
    resolved_mode = temp_req.resolve_essay_mode()

    # 0. Text Preprocessing Tahap Awal: Sanitasi HTML & Baris Baru (1x untuk referensi & tiap siswa)
    if actual_flags.clean_html:
        clean_reference_answers = [clean_html_and_newlines(ref) for ref in reference_answers]
        clean_student_answers = [clean_html_and_newlines(ans) for ans in student_answers]
        inner_flags = actual_flags.model_copy(update={"clean_html": False})
    else:
        clean_reference_answers = list(reference_answers)
        clean_student_answers = list(student_answers)
        inner_flags = actual_flags

    # 1. Jalankan Text Preprocessing, Synonym Expansion, Rule-Based & Math per siswa
    rule_scores: List[float] = []
    math_scores: List[float] = []
    all_matched: List[List[str]] = []
    factual_scores: List[float] = []

    for clean_ans in clean_student_answers:
        # Rule-Based (mencakup preprocessing teks & ekspansi sinonim kata kunci)
        r_score, matched = calculate_rule_based_score(
            clean_ans,
            keywords,
            required_keywords_count=required_keywords_count,
            flags=inner_flags,
        )
        rule_scores.append(r_score)
        all_matched.append(matched)

        # Evaluasi Matematis (jika soal memiliki math_steps)
        m_score = 0.0
        if math_steps:
            m_score = calculate_math_score(
                clean_ans,
                math_steps,
                reference_answers=clean_reference_answers,
                keywords=keywords,
            )
        math_scores.append(m_score)

        # Skor Faktual untuk pengecekan Tgate
        f_score = _compute_factual_score(
            rule_score=r_score,
            math_score=m_score,
            rule_weight=rule_weight,
            math_weight=math_weight,
            has_keywords=bool(keywords),
            has_answer=bool(clean_ans and clean_ans.strip()),
        )
        factual_scores.append(f_score)

    # 2. Pengecekan Tgate & Aktivasi LSA Sejati Berbasis Korpus
    lsa_scores = [0.0] * len(clean_student_answers)

    eligible_indices = [
        i for i, f_score in enumerate(factual_scores)
        if (f_score > 0) and (tgate == 0 or f_score >= tgate)
    ]

    if eligible_indices and lsa_weight > 0:
        eligible_answers = [clean_student_answers[i] for i in eligible_indices]
        batch_lsa_results = calculate_batch_lsa_scores(
            eligible_answers,
            clean_reference_answers,
            flags=inner_flags,
        )
        for idx_in_eligible, original_idx in enumerate(eligible_indices):
            lsa_scores[original_idx] = batch_lsa_results[idx_in_eligible]

    # 3. Agregasi Skor Akhir & Normalisasi terhadap max_score
    responses: List[GraderResponse] = []
    total_weights = rule_weight + lsa_weight + math_weight

    for i in range(len(clean_student_answers)):
        r_score = rule_scores[i]
        m_score = math_scores[i]
        l_score = lsa_scores[i]
        f_score = factual_scores[i]

        if (f_score > 0) and (tgate == 0 or f_score >= tgate):
            final_raw = (r_score * rule_weight) + (l_score * lsa_weight) + (m_score * math_weight)
        else:
            # Gagal Tgate -> return skor faktual
            final_raw = (r_score * rule_weight) + (m_score * math_weight)

        final_score = final_raw / total_weights if total_weights > 0 else 0.0
        scaled_score = (final_score / 100.0) * max_score

        responses.append(
            GraderResponse(
                final_score=round(final_score, 2),
                scaled_score=round(scaled_score, 2),
                rule_based_score=round(r_score, 2),
                lsa_score=round(l_score, 2),
                math_score=round(m_score, 2),
                matched_keywords=all_matched[i],
                essay_mode=resolved_mode,
            )
        )

    return responses


def _build_question_group_key(item: BatchEssayItem) -> Tuple[Any, ...]:
    """
    Menentukan kunci pengelompokan soal (grouping key):
    - Jika `id_soal` disertakan secara eksplisit, kelompokkan berdasarkan `id_soal`.
    - Jika `id_soal` tidak disertakan (legacy payload), kelompokkan berdasarkan
      identitas soal (reference_answers, keywords, math_steps, bobot, tgate, mode)
      agar jawaban-jawaban untuk soal yang sama tetap membentuk satu korpus LSA.
    """
    if item.id_soal is not None:
        return ("id_soal", str(item.id_soal))

    flags_dump = tuple(sorted(item.get_preprocessing_flags().model_dump().items()))
    kw_tuple = tuple((k.keyword, k.weight, k.strict) for k in item.keywords)
    math_tuple = tuple((m.equation, m.weight) for m in item.math_steps)
    ref_tuple = tuple(item.reference_answers)
    return (
        "signature",
        ref_tuple,
        kw_tuple,
        math_tuple,
        item.required_keywords_count,
        item.rule_weight,
        item.lsa_weight,
        item.math_weight,
        item.tgate,
        item.resolve_essay_mode(),
        flags_dump,
    )


def grade_essay_batch(batch_request: BatchEssayRequest) -> BatchGraderResponse:
    """
    Menilai sekumpulan jawaban esai seluruh siswa dari Sistem Informasi Asesmen:
    1. Menormalisasi input (mendukung flat `items` maupun `questions` + `answers`).
    2. Mengelompokkan seluruh jawaban siswa berdasarkan `id_soal`.
    3. Untuk setiap `id_soal`, menjalankan pipeline secara berurutan:
       Text Preprocessing -> Synonym Expand -> Rule-Based (& Math) -> Cek Tgate ->
       LSA Sejati berbasis Korpus Kelas -> Normalisasi berdasarkan `max_score`.
    4. Mengembalikan hasil penilaian sesuai `student_id` dan `id_soal` (serta `id`).
    """
    normalized_items = batch_request.to_normalized_items()

    # Kelompokkan indeks item berdasarkan id_soal (atau signature soal jika id_soal kosong)
    grouped_indices: Dict[Tuple[Any, ...], List[int]] = defaultdict(list)
    for idx, item in enumerate(normalized_items):
        key = _build_question_group_key(item)
        grouped_indices[key].append(idx)

    # Array penampung hasil sesuai urutan item masuk
    ordered_results: List[Optional[BatchGraderResultItem]] = [None] * len(normalized_items)

    # Proses grading per kelompok id_soal
    for _, indices in grouped_indices.items():
        first_item = normalized_items[indices[0]]
        group_student_answers = [normalized_items[i].student_answer for i in indices]

        graded_group = grade_batch_by_question(
            student_answers=group_student_answers,
            reference_answers=first_item.reference_answers,
            keywords=first_item.keywords,
            math_steps=first_item.math_steps,
            rule_weight=first_item.rule_weight,
            lsa_weight=first_item.lsa_weight,
            math_weight=first_item.math_weight,
            tgate=first_item.tgate,
            max_score=first_item.max_score,
            required_keywords_count=first_item.required_keywords_count,
            subject=first_item.subject,
            essay_mode=first_item.essay_mode,
            flags=first_item.get_preprocessing_flags(),
        )

        for pos_in_group, original_idx in enumerate(indices):
            orig_item = normalized_items[original_idx]
            graded_resp = graded_group[pos_in_group]

            # Jika max_score per item berbeda di dalam grup yang sama, sesuaikan scaled_score
            scaled = (
                round((graded_resp.final_score / 100.0) * orig_item.max_score, 2)
                if orig_item.max_score != first_item.max_score
                else graded_resp.scaled_score
            )

            resp_data = graded_resp.model_dump()
            resp_data["scaled_score"] = scaled

            ordered_results[original_idx] = BatchGraderResultItem(
                id=orig_item.id,
                id_soal=orig_item.id_soal,
                student_id=orig_item.student_id,
                student_name=orig_item.student_name,
                **resp_data,
            )

    final_results = [res for res in ordered_results if res is not None]
    return BatchGraderResponse(
        results=final_results,
        total_items=len(final_results),
    )