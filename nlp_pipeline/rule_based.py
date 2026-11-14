from typing import List, Tuple, Optional, Any

import re
from nlp_pipeline.preprocessing import preprocess_text, clean_html_and_newlines
from nlp_pipeline.synonym_expander import expand_keyword_variants
from schemas import KeywordItem


def _normalize_quotes(text: str) -> str:
    """Normalisasi kutip lengkung/tipografi (Word/HP) menjadi kutip ASCII standar."""
    if not text:
        return ""
    text = re.sub(r"[“”„‟]", '"', text)
    text = re.sub(r"[’‘´`]", "'", text)
    return text


def _prepare_strict_answer(student_answer_raw: str, flags: Optional[Any] = None) -> str:
    """
    Menyiapkan teks untuk pencocokan strict:
    - Tetap mempertahankan huruf kapital (case-sensitive) dan tanda baca.
    - Membersihkan tag HTML & HTML entities (&nbsp;, &quot;, dll.) jika clean_html aktif,
      sehingga teks seperti `<b>Jakarta</b>,` dapat dicocokkan oleh `**Jakarta,`.
    - Menormalisasi kutip lengkung ke kutip standar.
    """
    use_clean_html = True
    if flags is not None:
        if isinstance(flags, dict):
            use_clean_html = flags.get("clean_html", True)
        else:
            use_clean_html = getattr(flags, "clean_html", True)

    text = clean_html_and_newlines(student_answer_raw) if use_clean_html else (student_answer_raw or "")
    return _normalize_quotes(text)


def _build_strict_pattern(clean_variant: str) -> str:
    r"""
    Membangun pola regex strict dengan batas kata independen di sisi kiri & kanan.
    - Jika diawali karakter huruf/angka (\w), pasang (?<!\w) di kiri.
    - Jika diakhiri karakter huruf/angka (\w), pasang (?!\w) di kanan.
    Mengatasi kasus tanda baca menempel di ujung kata (mis. `**dan,` tidak akan
    salah cocok dengan `"badan,"`, dan `**"Halo"` tidak cocok dengan `"Halogen"`).
    """
    normalized_var = _normalize_quotes(clean_variant)
    pattern = re.escape(normalized_var)
    if re.match(r'^\w', normalized_var):
        pattern = r'(?<!\w)' + pattern
    if re.search(r'\w$', normalized_var):
        pattern = pattern + r'(?!\w)'
    return pattern


def _preprocess_preserving_stopwords(
    text: str,
    convert_numbers: bool = True,
    flags: Optional[Any] = None
) -> str:
    """Menjalankan preprocess_text tanpa menghapus stopwords sambil mempertahankan flag lainnya."""
    if flags is None:
        return preprocess_text(text, convert_numbers=convert_numbers, remove_stopwords=False)

    if hasattr(flags, "model_dump"):
        flag_dict = flags.model_dump()
    elif isinstance(flags, dict):
        flag_dict = dict(flags)
    else:
        flag_dict = dict(getattr(flags, "__dict__", {}))

    flag_dict["remove_stopwords"] = False
    return preprocess_text(text, convert_numbers=convert_numbers, flags=flag_dict)


def calculate_rule_based_score(
    student_answer_raw: str,
    keywords: List[KeywordItem],
    required_keywords_count: Optional[int] = None,
    convert_numbers: bool = True,
    flags: Optional[Any] = None
) -> Tuple[float, List[str]]:
    """
    Menghitung skor rule-based berdasarkan kemunculan keyword dengan bobot dinamis.

    Guru menentukan bobot masing-masing keyword sehingga kata kunci yang lebih
    penting memberikan kontribusi skor yang lebih besar.

    Mendukung 'strict' mode baik di level KeywordItem (`strict=True`) maupun per varian
    menggunakan awalan `**` di dalam pemisah `/` (selalu mengikat tepat 1 varian di kanannya).
    Kata kunci non-strict melindungi frasa yang mengandung stopword agar tidak terpotong.

    Jika 'required_keywords_count' diisi (misal: soal meminta minimal 5 dari 8 opsi):
    Sistem akan membagi matched_weight terhadap total bobot K kata kunci teratas,
    sehingga siswa yang berhasil memenuhi target minimal akan memperoleh nilai 100%.

    Mengembalikan:
        - skor (0-100)
        - list keyword original yang berhasil dicocokkan
    """
    if not keywords:
        return 0.0, []

    processed_answer = preprocess_text(student_answer_raw, convert_numbers=convert_numbers, flags=flags)
    processed_answer_with_stops = _preprocess_preserving_stopwords(
        student_answer_raw, convert_numbers=convert_numbers, flags=flags
    )
    strict_answer = _prepare_strict_answer(student_answer_raw, flags=flags)
    use_synonyms = getattr(flags, "use_synonyms", True) if flags is not None else True
    total_weight = sum(kw.weight for kw in keywords)

    if total_weight == 0:
        return 0.0, []

    matched: List[str] = []
    matched_weight = 0.0

    for kw_item in keywords:
        # Pisahkan keyword berdasarkan '/' atau ';' untuk menangani sinonim/alias
        variants = [k.strip() for k in re.split(r'[/;]', kw_item.keyword) if k.strip()]
        
        variant_matched = False
        matched_variant_text = ""
        non_strict_variants: List[str] = []

        for raw_variant in variants:
            # Strict per-varian: `**` mengikat tepat 1 varian di sebelah kanannya saja
            is_variant_strict = kw_item.strict or raw_variant.startswith('**')
            clean_variant = raw_variant[2:].strip() if raw_variant.startswith('**') else raw_variant
            if not clean_variant:
                continue

            if is_variant_strict:
                pattern = _build_strict_pattern(clean_variant)
                if re.search(pattern, strict_answer):
                    variant_matched = True
                    matched_variant_text = clean_variant
                    break  # Hentikan pencarian varian lain jika sudah ketemu
            else:
                non_strict_variants.append(clean_variant)
                # Proteksi Stopword pada Kata Kunci Non-Strict:
                # 1. Proses varian tanpa menghapus stopwords agar frasa kunci tetap utuh.
                variant_with_stops = _preprocess_preserving_stopwords(
                    clean_variant, convert_numbers=convert_numbers, flags=flags
                )
                variant_no_stops = preprocess_text(
                    clean_variant, convert_numbers=convert_numbers, flags=flags
                )

                if variant_with_stops:
                    pattern_with_stops = r'\b' + re.escape(variant_with_stops) + r'\b'
                    if re.search(pattern_with_stops, processed_answer_with_stops):
                        variant_matched = True
                        matched_variant_text = clean_variant
                        break

                # 2. Jika varian memang tidak mengandung stopword (sama setelah stopword removal),
                #    cocokkan juga ke processed_answer agar tetap toleran bila siswa menyelipkan kata penghubung.
                if variant_no_stops and variant_no_stops == variant_with_stops:
                    pattern_no_stops = r'\b' + re.escape(variant_no_stops) + r'\b'
                    if re.search(pattern_no_stops, processed_answer):
                        variant_matched = True
                        matched_variant_text = clean_variant
                        break

        # Jika belum cocok secara leksikal langsung dan ada varian non-strict + use_synonyms aktif,
        # coba cari melalui ekspansi sinonim otomatis (WordNet / Tesaurus Bilingual) khusus varian non-strict
        if not variant_matched and non_strict_variants and use_synonyms and processed_answer:
            expanded_pairs = expand_keyword_variants(non_strict_variants, flags=flags)
            for syn_processed, syn_label in expanded_pairs:
                if syn_processed:
                    pattern = r'\b' + re.escape(syn_processed) + r'\b'
                    if re.search(pattern, processed_answer) or re.search(pattern, processed_answer_with_stops):
                        variant_matched = True
                        matched_variant_text = syn_label
                        break


        # Jika salah satu varian dari keyword ini ditemukan, berikan skor
        if variant_matched:
            # Menyimpan kata asli yang diinputkan guru (misal: "wajib/harus") 
            # beserta varian yang berhasil di-match agar informatif
            matched.append(f"{kw_item.keyword} (matched: {matched_variant_text})")
            matched_weight += kw_item.weight

    # Hitung skor rule-based
    if required_keywords_count and required_keywords_count > 0:
        # Mengambil akumulasi bobot dari K kata kunci teratas sebagai target pembagi
        k = min(required_keywords_count, len(keywords))
        sorted_weights = sorted([kw.weight for kw in keywords], reverse=True)
        target_weight = sum(sorted_weights[:k])
        
        if target_weight > 0:
            score = min(100.0, (matched_weight / target_weight) * 100.0)
        else:
            score = 0.0
    else:
        score = (matched_weight / total_weight) * 100.0

    return score, matched
