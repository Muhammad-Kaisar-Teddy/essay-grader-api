import re
import sympy
from typing import List, Optional, Dict, Tuple, Set, Any
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    rationalize,
)
from schemas import MathStepItem, KeywordItem
from nlp_pipeline.preprocessing import clean_html_and_newlines


_RESERVED_MATH_FUNCS: Set[str] = {
    "sin", "cos", "tan", "asin", "acos", "atan",
    "sinh", "cosh", "tanh", "sqrt", "log", "ln", "exp", "abs", "pi",
}

# ==============================================================================
# KAMUS LABEL VARIABEL STANDAR BILINGUAL (AKTIF JIKA SIMBOL ADA DI math_steps)
# ==============================================================================
_STANDARD_SYMBOL_LABELS: Dict[str, Set[str]] = {
    "P": {"price", "harga", "tarif", "pe", "p_e"},
    "Q": {"quantity", "kuantitas", "jumlah", "jumlah barang", "pasokan", "qe", "q_e"},
    "v": {"velocity", "speed", "kecepatan", "kelajuan"},
    "t": {"time", "waktu", "durasi"},
    "s": {"distance", "displacement", "jarak", "perpindahan"},
    "m": {"mass", "massa"},
    "F": {"force", "gaya"},
}

# Kata pengisi / konektor di antara label verbal dan nilai numerik
_VERBAL_CONNECTORS = (
    r"(?:is|are|was|were|at|around|approximately|about|of|equals?|to|"
    r"adalah|ialah|yaitu|sebesar|senilai|sekitar|sama dengan|dengan)?"
)
_CURRENCY_TOKENS = r"(?:rp\.?|idr|\$|usd)?"
_UNIT_SUFFIXES = r"(?:unit|units|buah|pcs|rupiah)?"
_NUM_TOKEN_RE = r"-?\d+(?:\.\d{3})*(?:,\d+|\.\d+)?"


def _normalize_unicode_math(text: str) -> str:
    """Normalisasi karakter subscript Unicode, tanda minus tipografi, dan simbol panah."""
    if not text:
        return ""
    replacements = {
        "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
        "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
        "–": "-", "—": "-", "−": "-", "×": "*", "÷": "/",
        "≈": "=",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _normalize_number_str(num_str: str, scale_word: str = "") -> Optional[str]:
    """
    Mengonversi representasi angka Indonesia/Inggris (misal '30.000', '7,5', '4.5', '30 ribu')
    ke format float/int standar Python yang valid untuk SymPy.
    """
    if not num_str:
        return None
    s = num_str.strip()
    # Format ribuan Indonesia: 30.000 atau 24.000.000 -> hapus titik ribuan (abaikan desimal 0.xxx)
    s = re.sub(r"(?<!\b0)(?<=\d)\.(?=\d{3}(?!\d))", "", s)
    # Format desimal koma Indonesia: 7,5 atau 0,0005 -> titik desimal
    s = re.sub(r"(?<=\d),(?=\d)", ".", s)
    try:
        val = float(s)
    except ValueError:
        return None

    scale = scale_word.lower().strip()
    if scale in {"ribu", "rb", "thousand"}:
        val *= 1000
    elif scale in {"juta", "jt", "million"}:
        val *= 1_000_000

    if val.is_integer():
        return str(int(val))
    return str(val)


def _canonicalize_var_symbol(sym: str, expected_symbols: Set[str]) -> str:
    """
    Menormalkan alias variabel keseimbangan ('Pe' -> 'P', 'Qe' -> 'Q')
    ketika simbol kanonik ('P' atau 'Q') menjadi target evaluasi.
    """
    clean = sym.strip()
    if clean in {"Pe", "P_e", "pe", "p_e"} and "P" in expected_symbols:
        return "P"
    if clean in {"Qe", "Q_e", "qe", "q_e"} and "Q" in expected_symbols:
        return "Q"
    return clean


def _strip_question_prefixes(line: str) -> str:
    """Menghapus penomoran butir soal di awal baris (misal '1.', 'c.', 'c)', '(c)', 'Langkah 1:')."""
    line = re.sub(r"^\s*(?:(?:langkah|step)\s*)?(?:\d+[\.\:\)]|\(\d+\))\s*", "", line, flags=re.IGNORECASE)
    line = re.sub(r"^\s*(?:\([a-zA-Z]\)|[a-zA-Z][\.\)])\s+", "", line)
    return line.strip()


def _extract_expected_symbols(expected_steps: Optional[List[MathStepItem]]) -> Set[str]:
    """Mengumpulkan seluruh simbol variabel yang muncul pada expected math_steps."""
    symbols: Set[str] = set()
    if not expected_steps:
        return {"P", "Q"}
    for step in expected_steps:
        for tok in re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", step.equation or ""):
            if tok not in _RESERVED_MATH_FUNCS:
                symbols.add(tok)
    return symbols or {"P", "Q"}


def _build_math_context(
    expected_steps: Optional[List[MathStepItem]] = None,
    reference_answers: Optional[List[str]] = None,
    keywords: Optional[List[KeywordItem]] = None,
) -> Tuple[Set[str], Dict[str, str], Tuple[str, str]]:
    """
    Membangun konteks matematis soal secara dinamis dalam 3 tingkat prioritas:
    1. Auto-discovery dari kunci jawaban & keywords guru (mis. 'Harga (P)', 'Kecepatan (v)', 'Eq (Q:P)').
    2. Kamus label akademis standar yang diaktifkan hanya untuk simbol yang ada di expected_steps.
    3. Konvensi urutan koordinat default berdasarkan simbol target (mis. ('Q', 'P') untuk P & Q).
    """
    expected_symbols = _extract_expected_symbols(expected_steps)
    label_map: Dict[str, str] = {}
    coord_order: Optional[Tuple[str, str]] = None

    corpus_texts: List[str] = []
    if reference_answers:
        corpus_texts.extend(r for r in reference_answers if r)
    if keywords:
        corpus_texts.extend(k.keyword for k in keywords if getattr(k, "keyword", ""))
    if expected_steps:
        corpus_texts.extend(s.equation for s in expected_steps if getattr(s, "equation", ""))

    # --- Prioritas 1: Auto-discovery dari teks referensi dan kata kunci guru ---
    # Urutan koordinat: misal "Eq (Q:P)", "(Q ; P)", "E = (Q=7,5 ; P=4,5)", "(P, Q)"
    order_pattern = re.compile(
        r"\(\s*([A-Za-z][A-Za-z0-9_]?)\s*(?:=[^;,:)]*)?\s*[;:,]\s*([A-Za-z][A-Za-z0-9_]?)\s*(?:=[^;,:)]*)?\)"
    )
    paren_def_pattern = re.compile(r"\b([A-Za-z]{3,})\s*\(\s*([A-Za-z][A-Za-z0-9_]?)\s*\)")
    colon_def_pattern = re.compile(
        r"\b([A-Za-z]{3,})(?:\s+[A-Za-z]{3,})?\s*:\s*([A-Za-z][A-Za-z0-9_]?)\s*="
    )

    for text in corpus_texts:
        norm_text = _normalize_unicode_math(clean_html_and_newlines(text))
        for m in paren_def_pattern.finditer(norm_text):
            lbl = m.group(1).lower()
            sym = _canonicalize_var_symbol(m.group(2), expected_symbols)
            if sym in expected_symbols and lbl not in label_map:
                label_map[lbl] = sym

        for m in colon_def_pattern.finditer(norm_text):
            lbl = m.group(1).lower()
            sym = _canonicalize_var_symbol(m.group(2), expected_symbols)
            if sym in expected_symbols and lbl not in label_map:
                label_map[lbl] = sym

        if coord_order is None:
            for m in order_pattern.finditer(norm_text):
                s1 = _canonicalize_var_symbol(m.group(1), expected_symbols)
                s2 = _canonicalize_var_symbol(m.group(2), expected_symbols)
                if s1 != s2 and s1 in expected_symbols and s2 in expected_symbols:
                    coord_order = (s1, s2)
                    break

    # --- Prioritas 2: Kamus label akademis standar (hanya untuk simbol yang ada di expected_steps) ---
    for sym, default_labels in _STANDARD_SYMBOL_LABELS.items():
        if sym in expected_symbols:
            for lbl in default_labels:
                if lbl not in label_map:
                    label_map[lbl] = sym

    # --- Urutan koordinat default ---
    if coord_order is None:
        if "Q" in expected_symbols and "P" in expected_symbols:
            coord_order = ("Q", "P")
        elif "x" in expected_symbols and "y" in expected_symbols:
            coord_order = ("x", "y")
        else:
            sorted_syms = sorted(s for s in expected_symbols if len(s) <= 2)
            coord_order = (sorted_syms[0], sorted_syms[1]) if len(sorted_syms) >= 2 else ("Q", "P")

    return expected_symbols, label_map, coord_order


def _extract_coordinate_equations(
    text: str,
    default_order: Tuple[str, str],
    expected_symbols: Set[str],
) -> List[str]:
    """
    Mengekstrak pasangan koordinat non-persamaan seperti `eq (15 ; 30.000)`,
    `(P, Q) = (30.000, 15)`, `E (7,5 ; 4,5)`, `(15, 30.000)` menjadi bentuk persamaan `Q = 15`, `P = 30000`.
    """
    results: List[str] = []

    # 1. Cek pola dengan label sumbu eksplisit tepat sebelum koordinat: `(P, Q) = (30.000, 15)`
    labeled_coord_re = re.compile(
        r"\(\s*([A-Za-z][A-Za-z0-9_]?)\s*[;:,]\s*([A-Za-z][A-Za-z0-9_]?)\s*\)"
        r"\s*(?:=|adalah|is|:)?\s*"
        r"(?:[A-Za-z]{1,4}\s*)?"
        r"\(\s*(" + _NUM_TOKEN_RE + r")\s*(?:;|:\s*|,\s+)\s*(" + _NUM_TOKEN_RE + r")\s*\)",
        flags=re.IGNORECASE,
    )
    matched_spans: List[Tuple[int, int]] = []

    for m in labeled_coord_re.finditer(text):
        ax1 = _canonicalize_var_symbol(m.group(1), expected_symbols)
        ax2 = _canonicalize_var_symbol(m.group(2), expected_symbols)
        v1 = _normalize_number_str(m.group(3))
        v2 = _normalize_number_str(m.group(4))
        if v1 is not None and v2 is not None:
            results.append(f"{ax1} = {v1}")
            results.append(f"{ax2} = {v2}")
            matched_spans.append(m.span())

    # 2. Cek pola koordinat tanpa label sumbu eksplisit: `eq (15 ; 30.000)`, `(7,5 ; 4,5)`, `(15, 30.000)`
    # Hindari salah tangkap perkalian seperti `0,5 (30000)` dengan memastikan tidak didahului angka/operator langsung
    unlabeled_coord_re = re.compile(
        r"(?<![0-9\+\-\*\/])(?:\b(?:eq|equilibrium|titik(?:\s+keseimbangan)?|point|E)\b\s*(?:=|is|adalah|:)?\s*)?"
        r"\(\s*(" + _NUM_TOKEN_RE + r")\s*(?:;|:\s*|,\s+)\s*(" + _NUM_TOKEN_RE + r")\s*\)",
        flags=re.IGNORECASE,
    )

    for m in unlabeled_coord_re.finditer(text):
        # Lewati jika berada di dalam rentang yang sudah diproses sebagai koordinat berlabel
        if any(s <= m.start() and m.end() <= e for s, e in matched_spans):
            continue
        v1 = _normalize_number_str(m.group(1))
        v2 = _normalize_number_str(m.group(2))
        if v1 is not None and v2 is not None:
            ax1, ax2 = default_order
            results.append(f"{ax1} = {v1}")
            results.append(f"{ax2} = {v2}")

    return results


def _extract_verbal_equations(
    text: str,
    label_map: Dict[str, str],
) -> List[str]:
    """
    Mendeteksi frasa verbal (bilingual ID/EN) yang menyatakan nilai variabel tanpa '=',
    misal `equilibrium is at the price Rp. 30.000 with quantity of 15.` atau `Harga sebesar 30.000`.
    """
    if not label_map:
        return []

    results: List[str] = []
    sorted_labels = sorted(label_map.keys(), key=len, reverse=True)
    label_alt = "|".join(re.escape(lbl) for lbl in sorted_labels)

    # Cocokkan <label> + maksimal 3 kata pengisi/konektor/mata uang + <angka> + <skala/unit opsional>
    # Guard (?!\s*[\*\+\-\/A-Za-z]) memastikan ekspresi aljabar seperti `Harga = 2*Q` tidak dipotong jadi angka tunggal.
    verbal_re = re.compile(
        r"\b(" + label_alt + r")\b"
        r"(?:\s+" + _VERBAL_CONNECTORS + r"){0,3}"
        r"\s+" + _CURRENCY_TOKENS + r"\s*"
        r"(" + _NUM_TOKEN_RE + r")"
        r"(?:\s*(ribu|rb|thousand|juta|jt|million))?"
        r"(?:\s*" + _UNIT_SUFFIXES + r")"
        r"(?!\s*[\*\+\-\/A-Za-z])",
        flags=re.IGNORECASE,
    )

    for m in verbal_re.finditer(text):
        raw_label = m.group(1).lower()
        target_sym = label_map.get(raw_label)
        num_val = _normalize_number_str(m.group(2), m.group(3) or "")
        if target_sym and num_val is not None:
            results.append(f"{target_sym} = {num_val}")

    return results


def _split_line_equations(line: str) -> List[str]:
    """
    Memecah baris yang mengandung lebih dari satu persamaan (misal `Pe = 30.000 with the Qe = 15`
    atau persamaan berantai `Q = -60.000 / -4.000 = 15`) menjadi persamaan tunggal `LHS = RHS`.
    """
    eq_count = line.count("=")
    if eq_count == 0:
        return []
    if eq_count == 1:
        return [line]

    # 1. Jika ada >1 tanda '=', pisahkan penugasan variabel baru yang didahului angka/kata/kurung tutup
    #    (bukan didahului operator aritmatika seperti '+ Q =' pada '0.5P + Q = 10000').
    pre_split = re.sub(
        r"(?<=[\d\)a-zA-Z])\s+(?=[A-Za-z][A-Za-z0-9_]{0,2}\s*=)",
        "\n",
        line,
    )

    single_eqs: List[str] = []
    for sub in pre_split.split("\n"):
        if "=" not in sub:
            continue
        parts = [p.strip() for p in sub.split("=")]
        if len(parts) == 2:
            if parts[0] and parts[1]:
                single_eqs.append(f"{parts[0]} = {parts[1]}")
        else:
            # Persamaan berantai: A = B = C -> hasilkan (A = C), (A = B), dan (B = C)
            non_empty = [p for p in parts if p]
            if len(non_empty) >= 2:
                head = non_empty[0]
                for idx in range(1, len(non_empty)):
                    single_eqs.append(f"{head} = {non_empty[idx]}")
                    if idx + 1 < len(non_empty):
                        single_eqs.append(f"{non_empty[idx]} = {non_empty[idx + 1]}")

    return single_eqs


def _clean_single_equation(eq_line: str, expected_symbols: Set[str]) -> Optional[str]:
    """Membersihkan satu string persamaan mentah agar aman dibaca oleh SymPy parser."""
    line = _strip_question_prefixes(eq_line)

    # Normalisasi alias ekuilibrium Pe -> P dan Qe -> Q
    line = re.sub(r"\b(P_?e)\b", "P" if "P" in expected_symbols else r"\1", line, flags=re.IGNORECASE)
    line = re.sub(r"\b(Q_?e)\b", "Q" if "Q" in expected_symbols else r"\1", line, flags=re.IGNORECASE)

    # Hapus penanda mata uang dan satuan pendek (termasuk 'Rp' 2 huruf!)
    line = re.sub(r"\b(?:rp\.?|idr|usd|pcs|unit|units|buah)\b", "", line, flags=re.IGNORECASE)

    # Hapus kata teks biasa (>= 3 huruf) yang bukan simbol variabel target / fungsi matematis
    def _strip_non_symbol_words(match: re.Match) -> str:
        word = match.group(0)
        if word in expected_symbols or word in {"Qd", "Qs", "Pd", "Ps"} or word in _RESERVED_MATH_FUNCS:
            return word
        return ""

    cleaned = re.sub(r"\b[a-zA-Z]{3,}\b", _strip_non_symbol_words, line)

    # Hapus karakter non-matematis
    cleaned = re.sub(r"[^a-zA-Z0-9\+\-\*\/\=\(\)\.\,]", " ", cleaned)

    # Hapus koma yang tidak diapit angka (misal koma penutup kalimat "10.000, benar kan?")
    cleaned = re.sub(r"(?<!\d),|,(?!\d)", " ", cleaned)

    # Format angka (ribu vs desimal; pertahankan desimal 0.xxx)
    cleaned = re.sub(r"(?<!\b0)(?<=\d)\.(?=\d{3}(?!\d))", "", cleaned)
    cleaned = re.sub(r"(?<=\d),(?=\d)", ".", cleaned)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if "=" in cleaned:
        lhs, rhs = [p.strip() for p in cleaned.split("=", 1)]
        if lhs and rhs:
            return f"{lhs} = {rhs}"
    return None


def extract_math_equations(
    text: str,
    expected_steps: Optional[List[MathStepItem]] = None,
    reference_answers: Optional[List[str]] = None,
    keywords: Optional[List[KeywordItem]] = None,
) -> List[str]:
    """
    Mengekstrak seluruh persamaan matematis (eksplisit maupun non-persamaan hasil normalisasi
    koordinat & frasa verbal bilingual) agar siap dievaluasi oleh SymPy.
    """
    if not text:
        return []

    expected_symbols, label_map, coord_order = _build_math_context(
        expected_steps=expected_steps,
        reference_answers=reference_answers,
        keywords=keywords,
    )

    cleaned_raw = _normalize_unicode_math(clean_html_and_newlines(text))
    equations: List[str] = []
    seen: Set[str] = set()

    def _add_eq(eq: Optional[str]) -> None:
        if eq and eq not in seen:
            seen.add(eq)
            equations.append(eq)

    # 1. Ekstraksi pasangan koordinat non-persamaan & berlabel: `eq (15 ; 30.000)`, `(P, Q) = (30.000, 15)`
    for eq in _extract_coordinate_equations(cleaned_raw, coord_order, expected_symbols):
        _add_eq(eq)

    # 2. Ekstraksi frasa verbal bilingual (code-switching): `price Rp. 30.000 with quantity of 15`
    for eq in _extract_verbal_equations(cleaned_raw, label_map):
        _add_eq(eq)

    # 3. Ekstraksi persamaan eksplisit yang memiliki tanda '='
    normalized_lines_text = cleaned_raw.replace("=>", "\n")
    # Hanya pisahkan koma/titik koma di luar tanda kurung agar koordinat (Q=7,5 ; P=4,5) tidak rusak
    # maupun pemisah antar-persamaan seperti "P = 30.000; Q = 15" tetap terpisah bersih.
    normalized_lines_text = re.sub(r"[;]\s+", "\n", normalized_lines_text)
    normalized_lines_text = re.sub(r"(?<!\d),\s+(?=[A-Za-z])", "\n", normalized_lines_text)

    for raw_line in normalized_lines_text.split("\n"):
        if "=" not in raw_line:
            continue
        for single_raw in _split_line_equations(raw_line):
            _add_eq(_clean_single_equation(single_raw, expected_symbols))

    return equations


def preprocess_math_text(text: str) -> List[str]:
    """
    Ekstrak semua persamaan dari teks (eksplisit '=' maupun koordinat/verbal default P & Q).
    Dipertahankan sebagai antarmuka kompatibel untuk pemanggil langsung tanpa konteks eksternal.
    """
    return extract_math_equations(text)


def parse_equation(
    eq_str: str,
    extra_symbols: Optional[Set[str]] = None,
) -> Optional[sympy.Expr]:
    """
    Mengurai string persamaan (LHS = RHS) menjadi ekspresi SymPy (LHS - RHS).
    """
    if "=" not in eq_str:
        return None

    left_str, right_str = eq_str.split("=", 1)

    transformations = standard_transformations + (
        implicit_multiplication_application,
        rationalize,
    )

    # Q, S, D, E, I, N, O adalah kata/simbol khusus di SymPy; paksa menjadi Symbol biasa
    local_dict: Dict[str, Any] = {
        "Q": sympy.Symbol("Q"),
        "P": sympy.Symbol("P"),
        "Qd": sympy.Symbol("Qd"),
        "Qs": sympy.Symbol("Qs"),
        "Pd": sympy.Symbol("Pd"),
        "Ps": sympy.Symbol("Ps"),
        "S": sympy.Symbol("S"),
        "D": sympy.Symbol("D"),
        "E": sympy.Symbol("E"),
        "I": sympy.Symbol("I"),
        "N": sympy.Symbol("N"),
        "O": sympy.Symbol("O"),
    }
    if extra_symbols:
        for sym in extra_symbols:
            if (
                sym
                and sym not in local_dict
                and sym not in _RESERVED_MATH_FUNCS
                and re.match(r"^[A-Za-z][A-Za-z0-9_]*$", sym)
            ):
                local_dict[sym] = sympy.Symbol(sym)

    try:
        lhs = parse_expr(left_str, local_dict=local_dict, transformations=transformations)
        rhs = parse_expr(right_str, local_dict=local_dict, transformations=transformations)
        return lhs - rhs
    except Exception:
        return None


def calculate_math_score(
    student_answer: str,
    expected_steps: List[MathStepItem],
    reference_answers: Optional[List[str]] = None,
    keywords: Optional[List[KeywordItem]] = None,
) -> float:
    """
    Mengevaluasi jawaban matematika siswa terhadap langkah-langkah yang diharapkan.
    Mendukung pra-ekstraksi koordinat (Q, P)/(P, Q) dan frasa verbal bilingual
    berdasarkan konteks `expected_steps`, `reference_answers`, dan `keywords`.
    """
    if not expected_steps:
        return 0.0

    expected_symbols = _extract_expected_symbols(expected_steps)
    extracted_lines = extract_math_equations(
        student_answer,
        expected_steps=expected_steps,
        reference_answers=reference_answers,
        keywords=keywords,
    )

    student_exprs = []
    for line in extracted_lines:
        expr = parse_equation(line, extra_symbols=expected_symbols)
        if expr is not None:
            student_exprs.append(expr)

    total_weight = sum(step.weight for step in expected_steps)
    if total_weight == 0:
        return 0.0

    score = 0.0
    for step in expected_steps:
        expected_cleaned = extract_math_equations(
            step.equation,
            expected_steps=expected_steps,
            reference_answers=reference_answers,
            keywords=keywords,
        )
        if not expected_cleaned:
            continue

        expected_expr = parse_equation(expected_cleaned[0], extra_symbols=expected_symbols)
        if expected_expr is None:
            continue

        step_matched = False
        for s_expr in student_exprs:
            try:
                # Cek ekuivalensi (A = B sama dengan A - B = 0 atau B - A = 0)
                diff = sympy.simplify(s_expr - expected_expr)
                if diff == 0:
                    step_matched = True
                    break

                diff_reverse = sympy.simplify(s_expr + expected_expr)
                if diff_reverse == 0:
                    step_matched = True
                    break
            except Exception:
                continue

        if step_matched:
            score += step.weight

    return (score / total_weight) * 100.0
