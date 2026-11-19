# interactive_test.py

import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from schemas import EssayRequest, KeywordItem, MathStepItem
from grader import grade_essay

# Import fungsi internal untuk keperluan demo step-by-step
from nlp_pipeline.preprocessing import (
    clean_html_and_newlines, 
    _expand_contractions, 
    convert_numbers_to_words,
    strip_punctuation, 
    filter_numbers, 
    detect_language, 
    _normalize_word,
    preprocess_text,
    STOP_WORDS, 
    NEGATION_WORDS, 
    NUMBER_WORDS
)
from nlp_pipeline.rule_based import calculate_rule_based_score
from nlp_pipeline.math_evaluator import calculate_math_score, extract_math_equations
from nlp_pipeline.synonym_expander import align_tokens_for_lsa

def get_valid_float(prompt, default=None):
    while True:
        try:
            val = input(prompt).strip()
            if not val and default is not None:
                return default
            return float(val)
        except ValueError:
            print("❌ Input tidak valid. Harap masukkan angka (contoh: 0.5).")

def get_weights():
    print("\n=== TAHAP 1: KONFIGURASI BOBOT MODEL ===")
    print("Total bobot Rule, Math, dan LSA harus berjumlah 1.0 (100%).")
    
    while True:
        rule_weight = get_valid_float("Bobot Keyword (Rule-Based) [0.0 - 1.0]: ")
        math_weight = get_valid_float("Bobot Persamaan Matematika (Math) [0.0 - 1.0]: ", default=0.0)
        lsa_weight = get_valid_float("Bobot LSA [0.0 - 1.0]: ")
        
        total = rule_weight + math_weight + lsa_weight
        
        if abs(total - 1.0) < 1e-6:
            print(f"✅ Bobot valid: Rule={rule_weight}, Math={math_weight}, LSA={lsa_weight} (Total = {total})\n")
            return rule_weight, math_weight, lsa_weight
        else:
            print(f"❌ Total bobot saat ini {total:.2f}. Harus tepat 1.0. Silakan ulangi.\n")

def get_references():
    print("=== TAHAP 2: KUNCI JAWABAN (REFERENCE ANSWERS) ===")
    print("Masukkan kunci jawaban dari guru. Jika ada lebih dari satu, pisahkan dengan karakter '|'.")
    ref_input = input("Kunci Jawaban: ").strip()
    if not ref_input:
        print("❌ Kunci jawaban tidak boleh kosong!")
        sys.exit(1)
    
    references = [ref.strip() for ref in ref_input.split('|') if ref.strip()]
    print(f"✅ {len(references)} kunci jawaban diterima.\n")
    return references

def get_keywords():
    print("=== TAHAP 3: KATA KUNCI (KEYWORDS) ===")
    keywords = []
    add_kw = input("Apakah soal ini memiliki kata kunci? (y/n) [default: n]: ").strip().lower()
    if add_kw == 'y':
        while True:
            kw = input("Masukkan kata kunci (atau ketik 'selesai' untuk berhenti): ").strip()
            if kw.lower() == 'selesai' or not kw:
                break
            weight = get_valid_float(f"  Bobot untuk '{kw}' [default: 1.0]: ", default=1.0)
            keywords.append(KeywordItem(keyword=kw, weight=weight))
    print(f"✅ {len(keywords)} kata kunci dimasukkan.\n")
    return keywords

def get_math_steps():
    print("=== TAHAP 4: LANGKAH MATEMATIKA (MATH STEPS) ===")
    math_steps = []
    add_math = input("Apakah soal ini memiliki persamaan/langkah matematika? (y/n) [default: n]: ").strip().lower()
    if add_math == 'y':
        while True:
            eq = input("Masukkan persamaan (misal: 'P = 30000', atau ketik 'selesai'): ").strip()
            if eq.lower() == 'selesai' or not eq:
                break
            weight = get_valid_float(f"  Bobot untuk persamaan '{eq}' [default: 1.0]: ", default=1.0)
            math_steps.append(MathStepItem(equation=eq, weight=weight))
    print(f"✅ {len(math_steps)} langkah matematika dimasukkan.\n")
    return math_steps

# ==============================================================================
# FUNGSI DEMO: MENAMPILKAN PREPROCESSING STEP-BY-STEP
# ==============================================================================
def show_preprocessing_pipeline(text: str, flags):
    print("\n🔍 [1] TEXT PREPROCESSING PIPELINE:")
    print(f"  [Input Mentah]   : {text}")
    
    if flags.clean_html:
        text = clean_html_and_newlines(text)
        print(f"  0. Clean HTML    : {text}")
    
    detected_lang = detect_language(text)
    print(f"  1. Deteksi Bahasa: {detected_lang.upper()}")
    
    if flags.case_folding:
        text = text.lower()
        print(f"  2. Case Folding  : {text}")
        
    if flags.expand_contractions:
        text = _expand_contractions(text)
        print(f"  3. Contractions  : {text}")
        
    if flags.convert_numbers:
        text = convert_numbers_to_words(text, lang=detected_lang)
        print(f"  4. Num2Words     : {text}")
        
    if flags.remove_punctuation:
        text = strip_punctuation(text)
        print(f"  5. Hapus Tanda   : {text}")
    
    if not flags.preserve_years or flags.remove_punctuation:
        text = filter_numbers(text, preserve_years=flags.preserve_years)
        print(f"  6. Filter Angka  : {text}")
        
    tokens = text.split()
    if flags.remove_stopwords:
        tokens = [
            w for w in tokens
            if (w.isdigit() or w in NEGATION_WORDS or w in NUMBER_WORDS or w not in STOP_WORDS)
        ]
        print(f"  7. Hapus Stopword: {' '.join(tokens)}")
        
    if flags.use_lemmatization:
        tokens = [_normalize_word(t, lang=detected_lang) for t in tokens]
        print(f"  8. Lemmatization : {' '.join(tokens)}")
        
    print("-" * 50)

# ==============================================================================
# FUNGSI DEMO: PERHITUNGAN LSA (TF-IDF -> SVD -> COSINE)
# ==============================================================================
def show_lsa_process(student_ans: str, ref_ans: str, flags):
    print("\n🧮 [4] PROSES PERHITUNGAN LSA (TF-IDF -> SVD -> COSINE SIMILARITY):")
    
    # Preprocess
    proc_student = preprocess_text(student_ans, flags=flags)
    proc_ref = preprocess_text(ref_ans, flags=flags)
    
    use_synonyms = getattr(flags, "use_synonyms", True)
    if use_synonyms:
        aligned_student = align_tokens_for_lsa(proc_student, proc_ref)
        print(f"  - Penyelarasan Sinonim: '{proc_student}' -> '{aligned_student}'")
    else:
        aligned_student = proc_student
        
    docs = [proc_ref, aligned_student]
    print(f"  - Dokumen (Ref, Student): {docs}")

    # 1. TF-IDF
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    try:
        tfidf_matrix = vectorizer.fit_transform(docs)
        vocab = vectorizer.get_feature_names_out()
        print(f"  - 1. TF-IDF Matrix Shape: {tfidf_matrix.shape} (2 Dokumen x {len(vocab)} Fitur)")
        print(f"       Fitur (Vocabulary): {list(vocab)[:10]} {'...' if len(vocab) > 10 else ''}")
    except ValueError:
        print("  - ⚠️ TF-IDF Gagal (Tidak ada kata yang tersisa setelah preprocessing).")
        return 0.0

    # 2. SVD (Reduksi Dimensi)
    n_features = tfidf_matrix.shape[1]
    if n_features == 0:
        return 0.0
    k = max(1, min(tfidf_matrix.shape) - 1)
    print(f"  - 2. SVD Reduksi Dimensi (k={k}):")
    svd = TruncatedSVD(n_components=k, random_state=42)
    latent_matrix = svd.fit_transform(tfidf_matrix)
    print(f"       Matriks Laten (Dokumen x k):\n{np.round(latent_matrix, 4)}")

    # 3. Cosine Similarity
    sim = cosine_similarity(latent_matrix[0:1], latent_matrix[1:2])[0][0]
    sim = float(max(0.0, min(1.0, sim)))
    print(f"  - 3. Cosine Similarity di Ruang Laten: {sim:.4f}")
    
    # 4. Negation Penalty (Opsional untuk demo)
    from nlp_pipeline.lsa_model import _negation_penalty_factor
    neg_factor = _negation_penalty_factor(proc_student, proc_ref)
    print(f"  - 4. Negation Penalty Factor: {neg_factor:.4f}")
    
    final_lsa = sim * neg_factor * 100
    print(f"  - Skor LSA Akhir: ({sim:.4f} x {neg_factor:.4f}) x 100 = {final_lsa:.2f} / 100")
    print("-" * 50)
    return final_lsa

def run_interactive_test():
    print("="*70)
    print("🚀 SISTEM PENGUJIAN INTERAKTIF MODEL HIBRIDA (RULE + MATH + LSA)")
    print("="*70)

    rule_w, math_w, lsa_w = get_weights()
    references = get_references()
    keywords = get_keywords()
    math_steps = get_math_steps()
    tgate = get_valid_float("Tentukan Threshold Gate (Tgate) [0-100, default: 0]: ", default=0.0)
    max_score = get_valid_float("Tentukan Skor Maksimal Soal (max_score) [default: 100]: ", default=100.0)
    
    print("\n=== TAHAP 5: INPUT JAWABAN SISWA ===")
    while True:
        try:
            n = int(input("Berapa jumlah jawaban siswa yang ingin dievaluasi? (n): ").strip())
            if n > 0: break
        except ValueError:
            pass

    for i in range(1, n + 1):
        print("\n" + "="*70)
        print(f"📝 EVALUASI SISWA {i} DARI {n}")
        print("="*70)
        student_answer = input(f"Masukkan jawaban siswa {i}:\n> ").strip()
        
        if not student_answer:
            print("⚠️ Jawaban kosong, dilewati.")
            continue

        req = EssayRequest(
            student_answer=student_answer,
            reference_answers=references,
            keywords=keywords,
            math_steps=math_steps,
            rule_weight=rule_w,
            lsa_weight=lsa_w,
            math_weight=math_w,
            tgate=tgate,
            max_score=max_score
        )

        flags = req.get_preprocessing_flags()
        
        # 1. Tampilkan Preprocessing
        show_preprocessing_pipeline(student_answer, flags)

        # Persiapan variabel untuk perhitungan manual
        clean_student_answer = clean_html_and_newlines(student_answer) if flags.clean_html else student_answer
        clean_reference_answers = [clean_html_and_newlines(ref) for ref in references] if flags.clean_html else references
        inner_flags = flags.model_copy(update={"clean_html": False}) if flags.clean_html else flags

        # 2. Tampilkan Proses Rule-Based
        print("\n📑 [2] PERHITUNGAN RULE-BASED:")
        rule_score, matched = calculate_rule_based_score(
            clean_student_answer, keywords, req.required_keywords_count, flags=inner_flags
        )
        print(f"  - Kata Kunci Dicocokkan: {len(matched)} dari {len(keywords)}")
        for m in matched:
            print(f"    * {m}")
        
        # Hitung bobot total untuk demonstrasi
        total_weight = sum(kw.weight for kw in keywords)
        matched_weight = sum(kw.weight for kw in keywords if any(kw.keyword in m for m in matched))
        print(f"  - Bobot Total Keyword: {total_weight}")
        print(f"  - Bobot yang Cocok   : {matched_weight}")
        print(f"  - Rumus Skor         : ({matched_weight} / {total_weight}) * 100 = {rule_score:.2f}")
        print("-" * 50)

        # 3. Tampilkan Proses Math
        math_score = 0.0
        if math_steps:
            print("\n🔢 [3] PERHITUNGAN MATH EVALUATOR (SymPy):")
            extracted_eqs = extract_math_equations(
                clean_student_answer, expected_steps=math_steps, 
                reference_answers=clean_reference_answers, keywords=keywords
            )
            print(f"  - Persamaan yang diekstrak dari siswa: {extracted_eqs}")
            math_score = calculate_math_score(
                clean_student_answer, math_steps, 
                reference_answers=clean_reference_answers, keywords=keywords
            )
            print(f"  - Skor Math: {math_score:.2f} / 100")
            print("-" * 50)
        else:
            print("\n🔢 [3] PERHITUNGAN MATH EVALUATOR: (Tidak ada math_steps, dilewati)")
            print("-" * 50)

        # 4. Tampilkan Proses LSA
        best_lsa = 0.0
        if (rule_score * rule_w + math_score * math_w) / (rule_w + math_w) >= tgate or tgate == 0:
            print("\n🧠 [4] PERHITUNGAN LSA (Mengambil skor terbaik dari semua referensi):")
            for idx, ref in enumerate(clean_reference_answers):
                print(f"\n  -- Membandingkan dengan Referensi ke-{idx+1} --")
                lsa_val = show_lsa_process(clean_student_answer, ref, inner_flags)
                if lsa_val > best_lsa:
                    best_lsa = lsa_val
        else:
            print("\n🧠 [4] PERHITUNGAN LSA: DIMATIKAN (Skor Faktual di bawah Tgate)")
            print("-" * 50)

        # 5. Agregasi Skor Akhir
        print("\n⚖️ [5] AGREGASI SKOR AKHIR (FUSION):")
        factual_weight = rule_w + math_w
        factual_score = ((rule_score * rule_w) + (math_score * math_w)) / factual_weight if factual_weight > 0 else 0
        
        print(f"  - Skor Faktual (Rule + Math): {factual_score:.2f}")
        print(f"  - Tgate: {tgate}")
        
        if (factual_score > 0) and (tgate == 0 or factual_score >= tgate):
            print(f"  - Status Tgate: 🟢 LULUS (LSA Diaktifkan)")
            final_score_raw = (rule_score * rule_w) + (best_lsa * lsa_w) + (math_score * math_w)
            print(f"  - Rumus Fusion: (Rule * {rule_w}) + (LSA * {lsa_w}) + (Math * {math_w})")
            print(f"  - Perhitungan : ({rule_score:.2f} * {rule_w}) + ({best_lsa:.2f} * {lsa_w}) + ({math_score:.2f} * {math_w})")
        else:
            print(f"  - Status Tgate: 🔴 GAGAL (LSA Dimatikan)")
            final_score_raw = (rule_score * rule_w) + (math_score * math_w)
            print(f"  - Rumus Fusion: (Rule * {rule_w}) + (Math * {math_w})")
            print(f"  - Perhitungan : ({rule_score:.2f} * {rule_w}) + ({math_score:.2f} * {math_w})")
            
        final_score = final_score_raw / (rule_w + lsa_w + math_w)
        print(f"  - Final Score (0-100): {final_score:.2f}")
        
        # 6. Normalisasi ke Max Score
        scaled_score = (final_score / 100.0) * max_score
        print(f"  - Scaled Score (Max {max_score}): ({final_score:.2f} / 100) * {max_score} = {scaled_score:.2f}")

        # 7. VERIFIKASI DENGAN FUNGSI ASLI (grade_essay)
        print("\n✅ [6] VERIFIKASI HASIL DENGAN FUNGSI ASLI (grader.py):")
        try:
            real_result = grade_essay(req)
            print(f"  - Skor API (grade_essay) : {real_result.scaled_score:.2f} / {max_score}")
            print(f"  - Skor Manual (Layar)    : {scaled_score:.2f} / {max_score}")
            
            # Cek selisih (toleransi pembulatan 0.01)
            if abs(real_result.scaled_score - scaled_score) <= 0.01:
                print("  - Status: 🟢 COCOK (Logika manual = Logika grader.py)")
            else:
                print("  - Status: 🔴 BERBEDA (Ada ketidaksesuaian rumus!)")
        except Exception as e:
            print(f"  - ❌ Gagal memanggil grade_essay: {e}")

    print("\n" + "="*70)
    print("✅ PENGUJIAN SELESAI")
    print("="*70)

if __name__ == "__main__":
    try:
        run_interactive_test()
    except KeyboardInterrupt:
        print("\n\nPengujian dibatalkan oleh pengguna.")
        sys.exit(0)