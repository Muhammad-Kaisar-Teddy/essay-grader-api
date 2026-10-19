from typing import List, Tuple, Optional
import re
from nlp_pipeline.preprocessing import preprocess_text
from schemas import KeywordItem


def calculate_rule_based_score(
    student_answer_raw: str,
    keywords: List[KeywordItem],
    required_keywords_count: Optional[int] = None
) -> Tuple[float, List[str]]:
    """
    Menghitung skor rule-based berdasarkan kemunculan keyword dengan bobot dinamis.

    Guru menentukan bobot masing-masing keyword sehingga kata kunci yang lebih
    penting memberikan kontribusi skor yang lebih besar.

    Mendukung 'strict' mode: jika True, dicocokkan dengan teks raw (case-sensitive
    dan memperhatikan tanda baca). Jika False, menggunakan teks preprocessed.

    Jika 'required_keywords_count' diisi (misal: soal meminta minimal 5 dari 8 opsi):
    Sistem akan membagi matched_weight terhadap total bobot K kata kunci teratas,
    sehingga siswa yang berhasil memenuhi target minimal akan memperoleh nilai 100%.

    Mengembalikan:
        - skor (0-100)
        - list keyword original yang berhasil dicocokkan
    """
    if not keywords:
        return 0.0, []

    processed_answer = preprocess_text(student_answer_raw)
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

        for variant in variants:
            if kw_item.strict:
                pattern = re.escape(variant)
                if re.match(r'^\w+.*\w+$', variant) or re.match(r'^\w+$', variant):
                    pattern = r'\b' + pattern + r'\b'
                    
                if re.search(pattern, student_answer_raw):
                    variant_matched = True
                    matched_variant_text = variant
                    break # Hentikan pencarian varian lain jika sudah ketemu
            else:
                processed_variant = preprocess_text(variant)
                if processed_variant:
                    pattern = r'\b' + re.escape(processed_variant) + r'\b'
                    if re.search(pattern, processed_answer):
                        variant_matched = True
                        matched_variant_text = variant
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
