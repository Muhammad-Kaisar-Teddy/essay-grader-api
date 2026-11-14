import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to Hybrid Essay Grader API"}

# ---------------------------------------------------------------------------
# Test 1: Kasus dasar Bahasa Indonesia
# ---------------------------------------------------------------------------
def test_grade_basic_indonesian():
    payload = {
        "student_answer": "Fotosintesis adalah proses tumbuhan membuat makanan dengan bantuan sinar matahari",
        "reference_answers": [
            "Fotosintesis adalah proses di mana tumbuhan hijau membuat makanannya sendiri dari karbon dioksida dan air dengan menggunakan energi cahaya matahari."
        ],
        "keywords": [
            {"keyword": "tumbuhan", "weight": 1.0},
            {"keyword": "makanan", "weight": 1.5},
            {"keyword": "matahari", "weight": 2.0},
            {"keyword": "karbon dioksida", "weight": 2.0},
        ],
        "rule_weight": 0.4,
        "lsa_weight": 0.6,
    }
    response = client.post("/grade", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "final_score" in data
    assert "rule_based_score" in data
    assert "lsa_score" in data
    assert "matched_keywords" in data
    assert len(data["matched_keywords"]) > 0
    # Verifikasi fusion formula (toleransi 0.05 untuk floating-point rounding)
    expected = (data["rule_based_score"] * 0.4) + (data["lsa_score"] * 0.6)
    assert data["final_score"] == pytest.approx(expected, abs=0.05)


# ---------------------------------------------------------------------------
# Test 2: Multiple Reference Answers (bilingual)
# ---------------------------------------------------------------------------
def test_grade_multiple_reference_answers_bilingual():
    payload = {
        "student_answer": "Photosynthesis is the process where plants produce food using sunlight and carbon dioxide.",
        "reference_answers": [
            "Fotosintesis adalah proses tumbuhan hijau membuat makanan dari karbon dioksida dan air menggunakan energi cahaya matahari.",
            "Photosynthesis is the process by which green plants make their own food from carbon dioxide and water using sunlight energy.",
        ],
        "keywords": [
            {"keyword": "photosynthesis", "weight": 2.0},
            {"keyword": "plants", "weight": 1.0},
            {"keyword": "sunlight", "weight": 1.5},
            {"keyword": "carbon dioxide", "weight": 2.0},
        ],
        "rule_weight": 0.5,
        "lsa_weight": 0.5,
    }
    response = client.post("/grade", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["final_score"] > 0
    assert data["lsa_score"] > 0


# ---------------------------------------------------------------------------
# Test 3: Penanganan Negasi via N-gram
# ---------------------------------------------------------------------------
def test_negation_lowers_lsa_score():
    base_payload = {
        "reference_answers": ["Suhu air naik ketika dipanaskan."],
        "keywords": [{"keyword": "suhu", "weight": 1.0}],
        "rule_weight": 0.0,
        "lsa_weight": 1.0,
    }

    correct = {**base_payload, "student_answer": "Suhu air naik saat dipanaskan."}
    negated = {**base_payload, "student_answer": "Suhu air tidak naik saat dipanaskan."}

    r_correct = client.post("/grade", json=correct)
    r_negated = client.post("/grade", json=negated)
    
    score_correct = r_correct.json()["lsa_score"]
    score_negated = r_negated.json()["lsa_score"]

    assert score_correct > score_negated, (
        f"Expected correct ({score_correct}) > negated ({score_negated})"
    )


# ---------------------------------------------------------------------------
# Test 4: Bobot dinamis keyword
# ---------------------------------------------------------------------------
def test_dynamic_keyword_weights():
    payload_low_weight = {
        "student_answer": "tumbuhan melakukan proses",
        "reference_answers": ["Fotosintesis adalah proses tumbuhan menghasilkan makanan."],
        "keywords": [
            {"keyword": "tumbuhan", "weight": 1.0},
            {"keyword": "makanan", "weight": 5.0},
        ],
        "rule_weight": 1.0,
        "lsa_weight": 0.0,
    }
    payload_high_weight = {
        "student_answer": "makanan dihasilkan dari proses ini",
        "reference_answers": ["Fotosintesis adalah proses tumbuhan menghasilkan makanan."],
        "keywords": [
            {"keyword": "tumbuhan", "weight": 1.0},
            {"keyword": "makanan", "weight": 5.0},
        ],
        "rule_weight": 1.0,
        "lsa_weight": 0.0,
    }

    r_low = client.post("/grade", json=payload_low_weight)
    r_high = client.post("/grade", json=payload_high_weight)

    assert r_high.json()["rule_based_score"] > r_low.json()["rule_based_score"]

# ---------------------------------------------------------------------------
# Test 5: Strict Keyword (Case Sensitive & Punctuation)
# ---------------------------------------------------------------------------
def test_strict_keyword_matching():
    payload = {
        "reference_answers": ["Ibu kota Indonesia adalah Jakarta."],
        "keywords": [
            {"keyword": "Jakarta", "weight": 1.0, "strict": True}, # Harus 'J' besar
            {"keyword": "ibu kota", "weight": 1.0, "strict": False} # Bebas
        ],
        "rule_weight": 1.0,
        "lsa_weight": 0.0,
    }
    
    # 1. Jawaban benar (kapital sesuai)
    p_correct = {**payload, "student_answer": "Ibu kota kita adalah Jakarta."}
    r_correct = client.post("/grade", json=p_correct).json()
    assert "Jakarta (matched: Jakarta)" in r_correct["matched_keywords"]
    assert "ibu kota (matched: ibu kota)" in r_correct["matched_keywords"]
    assert r_correct["rule_based_score"] == 100.0
    
    # 2. Jawaban salah (huruf kecil)
    p_wrong_case = {**payload, "student_answer": "Ibu kota kita adalah jakarta."}
    r_wrong_case = client.post("/grade", json=p_wrong_case).json()
    assert "Jakarta (matched: Jakarta)" not in r_wrong_case["matched_keywords"] # Gagal match karena strict
    assert "ibu kota (matched: ibu kota)" in r_wrong_case["matched_keywords"] # Tetap match karena tidak strict
    assert r_wrong_case["rule_based_score"] == 50.0

# ---------------------------------------------------------------------------
# Test 6: Strict Keyword (Tanda Koma di Tengah)
# ---------------------------------------------------------------------------
def test_strict_keyword_punctuation():
    payload = {
        "reference_answers": ["di sungai, pasar, dan lingkungan permukiman"],
        "keywords": [
            {"keyword": "sungai, pasar, dan", "weight": 1.0, "strict": True}, 
        ],
        "rule_weight": 1.0,
        "lsa_weight": 0.0,
    }
    
    # 1. Jawaban dengan koma yang benar
    p_correct = {**payload, "student_answer": "Mereka ada di sungai, pasar, dan lingkungan."}
    r_correct = client.post("/grade", json=p_correct).json()
    assert "sungai, pasar, dan (matched: sungai, pasar, dan)" in r_correct["matched_keywords"]
    assert r_correct["rule_based_score"] == 100.0
    
    # 2. Jawaban dengan koma yang salah (kurang koma sebelum 'dan')
    p_wrong = {**payload, "student_answer": "Mereka ada di sungai, pasar dan lingkungan."}
    r_wrong = client.post("/grade", json=p_wrong).json()
    assert "sungai, pasar, dan (matched: sungai, pasar, dan)" not in r_wrong["matched_keywords"]
    assert r_wrong["rule_based_score"] == 0.0

# ---------------------------------------------------------------------------
# Test 7: Keyword Alias (Synonym Groups with '/')
# ---------------------------------------------------------------------------
def test_keyword_alias_synonyms():
    payload = {
        "reference_answers": ["Siswa harus sadar akan kewajiban mereka."],
        "keywords": [
            {"keyword": "kewajiban / tanggung jawab / tugas", "weight": 2.0},
            {"keyword": "sadar", "weight": 1.0}
        ],
        "rule_weight": 1.0,
        "lsa_weight": 0.0,
    }

    # 1. Menggunakan kata asli 'kewajiban'
    p1 = {**payload, "student_answer": "Mereka harus sadar akan kewajiban."}
    r1 = client.post("/grade", json=p1).json()
    assert r1["rule_based_score"] == 100.0
    assert "kewajiban / tanggung jawab / tugas (matched: kewajiban)" in r1["matched_keywords"]

    # 2. Menggunakan alias 'tanggung jawab'
    p2 = {**payload, "student_answer": "Mereka harus sadar akan tanggung jawab."}
    r2 = client.post("/grade", json=p2).json()
    assert r2["rule_based_score"] == 100.0
    assert "kewajiban / tanggung jawab / tugas (matched: tanggung jawab)" in r2["matched_keywords"]

    # 3. Menggunakan alias 'tugas' dan kehilangan kata 'sadar'
    p3 = {**payload, "student_answer": "Ini adalah tugas mereka."}
    r3 = client.post("/grade", json=p3).json()
    # Hanya match alias tugas (bobot 2.0), gagal sadar (bobot 1.0). Skor = (2/3)*100 = 66.66
    assert r3["rule_based_score"] == pytest.approx(66.66, abs=0.1)
    assert "kewajiban / tanggung jawab / tugas (matched: tugas)" in r3["matched_keywords"]

# ---------------------------------------------------------------------------
# Test 8: Math Evaluation with SymPy
# ---------------------------------------------------------------------------
def test_math_evaluation_sympy():
    payload = {
        "student_answer": "Diketahui bahwa rumus awal adalah Q = 10000 - 0.5P => maka dari itu",
        "reference_answers": ["Q = 10000 - 0.5P"],
        "keywords": [],
        "math_steps": [
            {"equation": "Q = 10000 - 0.5 * P", "weight": 1.0}
        ],
        "rule_weight": 0.0,
        "lsa_weight": 0.0,
        "math_weight": 1.0,
    }

    # 1. Exact mathematical match
    r1 = client.post("/grade", json=payload).json()
    assert r1["math_score"] == 100.0
    assert r1["final_score"] == 100.0

    # 2. Equivalent mathematical formulation (swapped sides, different coefficient style)
    p2 = {**payload, "student_answer": "Ternyata hasilnya 0.5P + Q = 10.000, benar kan?"}
    r2 = client.post("/grade", json=p2).json()
    assert r2["math_score"] == 100.0

    # 3. Incorrect math
    p3 = {**payload, "student_answer": "Q = 5000 - 0.5P"}
    r3 = client.post("/grade", json=p3).json()
    assert r3["math_score"] == 0.0


# ---------------------------------------------------------------------------
# Test 9: Required Keywords Count (Target Match Capping)
# ---------------------------------------------------------------------------
def test_required_keywords_count():
    # Ada 8 opsi kata kunci, tapi soal hanya meminta minimal 3 kata kunci untuk nilai 100%
    payload = {
        "student_answer": "Siswa menyebutkan apel, jeruk, dan mangga saja.",
        "reference_answers": ["Siswa menyebutkan buah-buahan lokal seperti apel, jeruk, mangga, pisang, pepaya, nanas, melon, semangka."],
        "keywords": [
            {"keyword": "apel", "weight": 1.0},
            {"keyword": "jeruk", "weight": 1.0},
            {"keyword": "mangga", "weight": 1.0},
            {"keyword": "pisang", "weight": 1.0},
            {"keyword": "pepaya", "weight": 1.0},
            {"keyword": "nanas", "weight": 1.0},
            {"keyword": "melon", "weight": 1.0},
            {"keyword": "semangka", "weight": 1.0},
        ],
        "required_keywords_count": 3,
        "rule_weight": 1.0,
        "lsa_weight": 0.0,
        "tgate": 0.0
    }

    # 1. Siswa menjawab 3 dari 8 kata kunci -> berhak dapat nilai 100%
    r1 = client.post("/grade", json=payload).json()
    assert r1["rule_based_score"] == 100.0
    assert len(r1["matched_keywords"]) == 3

    # 2. Siswa menjawab 4 dari 8 kata kunci -> tetap 100% (capping)
    p2 = {**payload, "student_answer": "Siswa menyebutkan apel, jeruk, mangga, dan pisang."}
    r2 = client.post("/grade", json=p2).json()
    assert r2["rule_based_score"] == 100.0
    assert len(r2["matched_keywords"]) == 4

    # 3. Siswa hanya menjawab 2 dari target 3 -> skor = (2/3) * 100 = 66.67
    p3 = {**payload, "student_answer": "Hanya ada apel dan jeruk di sini."}
    r3 = client.post("/grade", json=p3).json()
    assert r3["rule_based_score"] == pytest.approx(66.67, abs=0.1)
    assert len(r3["matched_keywords"]) == 2


# ---------------------------------------------------------------------------
# Test 10: Normalisasi HTML, Enter (<p>), dan Shift + Enter (<br>) pada Math
# ---------------------------------------------------------------------------
def test_math_steps_with_html_enter_and_shift_enter():
    base_payload = {
        "reference_answers": ["Q = 10000 - 0.5P\nP = 20000 - 2Q"],
        "keywords": [],
        "math_steps": [
            {"equation": "Q = 10000 - 0.5 * P", "weight": 1.0},
            {"equation": "P = 20000 - 2 * Q", "weight": 1.0}
        ],
        "rule_weight": 0.0,
        "lsa_weight": 0.0,
        "math_weight": 1.0,
    }

    # 1. Siswa menggunakan Enter di Rich Text Editor (<p>...</p>)
    p_enter = {
        **base_payload,
        "student_answer": "<p>Langkah 1: Q = 10000 - 0.5P</p><p>Langkah 2: P = 20000 - 2Q</p>"
    }
    r_enter = client.post("/grade", json=p_enter).json()
    assert r_enter["math_score"] == 100.0, f"Expected 100.0 for Enter (<p>), got {r_enter['math_score']}"

    # 2. Siswa menggunakan Shift + Enter di Rich Text Editor (<br> / <br/>)
    p_shift_enter = {
        **base_payload,
        "student_answer": "Langkah 1: Q = 10000 - 0.5P<br>Langkah 2: P = 20000 - 2Q"
    }
    r_shift_enter = client.post("/grade", json=p_shift_enter).json()
    assert r_shift_enter["math_score"] == 100.0, f"Expected 100.0 for Shift+Enter (<br>), got {r_shift_enter['math_score']}"

    # 3. Siswa menggunakan Enter biasa pada Textarea (\r\n / \n)
    p_crlf = {
        **base_payload,
        "student_answer": "Langkah 1: Q = 10000 - 0.5P\r\nLangkah 2: P = 20000 - 2Q"
    }
    r_crlf = client.post("/grade", json=p_crlf).json()
    assert r_crlf["math_score"] == 100.0, f"Expected 100.0 for CRLF (\\r\\n), got {r_crlf['math_score']}"


# ---------------------------------------------------------------------------
# Test 11: Normalisasi HTML pada Jawaban Esai Deskriptif & Strict Keyword
# ---------------------------------------------------------------------------
def test_essay_with_rich_text_html_entities_and_strict():
    payload = {
        "student_answer": "<p>Ibu kota kita adalah <b>Jakarta</b>&nbsp;dan memiliki <i>sungai</i>.</p>",
        "reference_answers": ["<p>Ibu kota kita adalah <b>Jakarta</b>.</p>"],
        "keywords": [
            {"keyword": "Jakarta", "weight": 1.0, "strict": True},
            {"keyword": "sungai", "weight": 1.0, "strict": False}
        ],
        "rule_weight": 0.5,
        "lsa_weight": 0.5,
    }
    r = client.post("/grade", json=payload).json()
    assert r["rule_based_score"] == 100.0
    assert "Jakarta (matched: Jakarta)" in r["matched_keywords"]
    assert r["lsa_score"] > 0


# ---------------------------------------------------------------------------
# Test 12: Unit Test Ekspansi Kontraksi Bahasa Inggris
# ---------------------------------------------------------------------------
def test_expand_contractions_unit():
    from nlp_pipeline.preprocessing import expand_contractions
    assert expand_contractions("don't") == "do not"
    assert expand_contractions("won't") == "will not"
    assert expand_contractions("can't") == "can not"
    assert expand_contractions("it's") == "it is"
    assert expand_contractions("they're") == "they are"
    # Curly apostrophe (’ dan ‘)
    assert expand_contractions("don’t") == "do not"
    assert expand_contractions("it’s") == "it is"
    # Tanpa apostrof
    assert expand_contractions("cant") == "can not"
    assert expand_contractions("wont") == "will not"


# ---------------------------------------------------------------------------
# Test 13: Integrasi Kontraksi pada Evaluasi Esai Bahasa Inggris (Rule-Based & LSA)
# ---------------------------------------------------------------------------
def test_grade_english_contractions_and_negation_symmetry():
    # Kunci memakai "does not", siswa memakai "doesn't" dan kutip keriting "doesn’t"
    payload = {
        "reference_answers": ["The water temperature does not increase when boiling."],
        "keywords": [
            {"keyword": "not increase / does not increase", "weight": 2.0},
            {"keyword": "temperature", "weight": 1.0}
        ],
        "rule_weight": 0.5,
        "lsa_weight": 0.5,
    }

    # 1. Siswa menjawab memakai kontraksi standar ("doesn't")
    p1 = {**payload, "student_answer": "The water temperature doesn't increase when boiling."}
    r1 = client.post("/grade", json=p1).json()
    assert r1["rule_based_score"] == 100.0
    assert r1["lsa_score"] > 80.0  # Kemiripan tinggi dan tidak terkena penalti asimetri negasi

    # 2. Siswa menjawab memakai curly apostrophe ("doesn’t")
    p2 = {**payload, "student_answer": "The water temperature doesn’t increase when boiling."}
    r2 = client.post("/grade", json=p2).json()
    assert r2["rule_based_score"] == 100.0
    assert r2["lsa_score"] > 80.0


# ---------------------------------------------------------------------------
# Test 14: Unit Test Konversi Angka ke Kata Selektif (Num2Words)
# ---------------------------------------------------------------------------
def test_convert_numbers_to_words_unit():
    from nlp_pipeline.preprocessing import convert_numbers_to_words
    # Bahasa Indonesia
    assert "lima sila" in convert_numbers_to_words("Pancasila memiliki 5 sila dan 3 pilar.")
    assert "tiga pilar" in convert_numbers_to_words("Pancasila memiliki 5 sila dan 3 pilar.")
    # Bahasa Inggris otomatis
    assert "three main factors" in convert_numbers_to_words("There are 3 main factors and 10 principles.")
    assert "ten principles" in convert_numbers_to_words("There are 3 main factors and 10 principles.")
    # Selektif: mengabaikan baris persamaan matematika dengan tanda '='
    math_line = "Persamaan: 2x + 4 = 10"
    assert convert_numbers_to_words(math_line) == math_line
    # Selektif: desimal dan angka tahun tidak diubah menjadi kata panjang
    assert "4.0" in convert_numbers_to_words("Era Revolusi Industri 4.0 terjadi tahun 2024.")
    assert "2024" in convert_numbers_to_words("Era Revolusi Industri 4.0 terjadi tahun 2024.")


# ---------------------------------------------------------------------------
# Test 15: Integrasi Num2Words pada Evaluasi Esai (Angka vs Huruf)
# ---------------------------------------------------------------------------
def test_grade_essay_with_num2words_matching():
    # Kunci memakai kata "lima sila", siswa menulis angka "5 sila"
    payload = {
        "student_answer": "Negara Indonesia berlandaskan 5 sila dan 3 pilar utama.",
        "reference_answers": ["Negara Indonesia berlandaskan lima sila dan tiga pilar utama."],
        "keywords": [
            {"keyword": "lima sila", "weight": 2.0},
            {"keyword": "tiga pilar", "weight": 1.0}
        ],
        "rule_weight": 0.5,
        "lsa_weight": 0.5,
    }
    r = client.post("/grade", json=payload).json()
    # Siswa tetap dapat nilai sempurna meskipun menulis "5" bukan "lima"
    assert r["rule_based_score"] == 100.0
    assert r["lsa_score"] > 85.0
    assert "lima sila (matched: lima sila)" in r["matched_keywords"]
    assert "tiga pilar (matched: tiga pilar)" in r["matched_keywords"]


# ---------------------------------------------------------------------------
# Test 16: Opsi Menonaktifkan Num2Words (convert_numbers=False)
# ---------------------------------------------------------------------------
def test_grade_essay_disable_num2words():
    # Jika convert_numbers dinonaktifkan, "5 sila" tidak akan dikonversi ke "lima sila"
    payload = {
        "student_answer": "Negara Indonesia berlandaskan 5 sila.",
        "reference_answers": ["Negara Indonesia berlandaskan lima sila."],
        "keywords": [
            {"keyword": "lima sila", "weight": 2.0}
        ],
        "convert_numbers": False,
        "rule_weight": 1.0,
        "lsa_weight": 0.0,
    }
    r = client.post("/grade", json=payload).json()
    # Angka 5 dihapus oleh regex pembersih angka biasa tanpa konversi, sehingga gagal match
    assert r["rule_based_score"] == 0.0


# ---------------------------------------------------------------------------
# Test 17: Preservasi Angka Tahun 4-Digit (Sejarah / Faktual)
# ---------------------------------------------------------------------------
def test_preserve_historical_years():
    base_payload = {
        "reference_answers": ["Proklamasi kemerdekaan Indonesia dibacakan pada 17 Agustus 1945."],
        "keywords": [
            {"keyword": "1945", "weight": 2.0},
            {"keyword": "agustus", "weight": 1.0}
        ],
        "rule_weight": 0.5,
        "lsa_weight": 0.5,
    }

    # 1. Siswa menjawab dengan tahun yang benar ("1945")
    p_correct = {**base_payload, "student_answer": "Indonesia merdeka pada 17 Agustus 1945."}
    r_correct = client.post("/grade", json=p_correct).json()
    assert r_correct["rule_based_score"] == 100.0
    assert "1945 (matched: 1945)" in r_correct["matched_keywords"]
    assert r_correct["lsa_score"] > 50.0

    # 2. Siswa menjawab dengan tahun yang salah ("1948")
    p_wrong = {**base_payload, "student_answer": "Indonesia merdeka pada 17 Agustus 1948."}
    r_wrong = client.post("/grade", json=p_wrong).json()
    # Keyword '1945' (bobot 2.0) gagal cocok, hanya 'agustus' (bobot 1.0) yang cocok -> skor (1/3)*100 = 33.33
    assert r_wrong["rule_based_score"] == pytest.approx(33.33, abs=0.1)
    assert "1945 (matched: 1945)" not in r_wrong["matched_keywords"]
    assert "agustus (matched: agustus)" in r_wrong["matched_keywords"]
    # Siswa dengan tahun benar mendapat skor LSA lebih tinggi daripada siswa yang salah tahun
    assert r_correct["lsa_score"] > r_wrong["lsa_score"]


# ---------------------------------------------------------------------------
# Test 18: Lemmatisasi Bilingual (Bahasa Indonesia & Bahasa Inggris)
# ---------------------------------------------------------------------------
def test_bilingual_lemmatization():
    from nlp_pipeline.preprocessing import _normalize_word, lemmatize_text

    # 1. Lemmatisasi Bahasa Indonesia (menggunakan nlp-id)
    assert _normalize_word("menuliskan") == "tulis"
    assert _normalize_word("tumbuhan") == "tumbuh"
    assert _normalize_word("makanan") == "makan"
    assert _normalize_word("kewajiban") == "wajib"
    assert _normalize_word("mengembangkan") == "kembang"

    # 2. Lemmatisasi Bahasa Inggris (menggunakan spaCy en_core_web_sm)
    assert _normalize_word("running") == "run"
    assert _normalize_word("producing") == "produce"
    assert _normalize_word("plants") == "plant"
    assert _normalize_word("children") == "child"
    assert _normalize_word("books") == "book"

    # 3. Kata yang HARUS dipertahankan (negasi, bilangan, angka tahun)
    assert _normalize_word("tidak") == "tidak"
    assert _normalize_word("bukan") == "bukan"
    assert _normalize_word("not") == "not"
    assert _normalize_word("lima") == "lima"
    assert _normalize_word("five") == "five"
    assert _normalize_word("1945") == "1945"
    assert _normalize_word("2024") == "2024"

    # 4. Lemmatisasi kalimat utuh (Code-switching / Campuran)
    mixed_sentence = "Siswa sedang running algoritma machine learning di laptop"
    lemmatized = lemmatize_text(mixed_sentence)
    # Memastikan kata bahasa inggris "running" tereduksi jadi "run", dan kata ID tereduksi
    assert "run" in lemmatized
    assert "laptop" in lemmatized


# ---------------------------------------------------------------------------
# Test 19: Part-of-Speech (POS) Tagging Bilingual
# ---------------------------------------------------------------------------
def test_bilingual_pos_tagging():
    from nlp_pipeline.preprocessing import get_pos_tags

    # 1. POS Tagging Bahasa Indonesia
    id_text = "Tumbuhan hijau membuat makanannya sendiri."
    id_tags = get_pos_tags(id_text)
    assert len(id_tags) > 0
    tokens_dict = dict(id_tags)
    assert tokens_dict.get("Tumbuhan") == "NN"
    assert tokens_dict.get("hijau") == "JJ"
    assert tokens_dict.get("membuat") == "VB"

    # 2. POS Tagging Bahasa Inggris
    en_text = "Green plants produce food using sunlight."
    en_tags = get_pos_tags(en_text)
    assert len(en_tags) > 0
    en_tokens_dict = dict(en_tags)
    assert en_tokens_dict.get("plants") == "NOUN"
    assert en_tokens_dict.get("produce") == "VERB"
    assert en_tokens_dict.get("food") == "NOUN"

    # 3. POS Tagging Campuran (Code-Switching)
    mixed_text = "Siswa sedang running machine learning di laptop."
    mixed_tags = get_pos_tags(mixed_text, refine_foreign=False)
    # Kata bahasa inggris dalam kalimat Indonesia diberi tag 'FW' (Foreign Word)
    fw_tokens = [tok for tok, tag in mixed_tags if tag == "FW"]
    assert "running" in fw_tokens or "machine" in fw_tokens

    # 4. Refined Foreign Words (refine_foreign=True)
    refined_tags = get_pos_tags(mixed_text, refine_foreign=True)
    tag_lookup = dict(refined_tags)
    assert "FW (VERB)" in tag_lookup.get("running", "")


# ---------------------------------------------------------------------------
# Test 20: Endpoint API /pos-tag dan /lemmatize
# ---------------------------------------------------------------------------
def test_api_pos_tag_and_lemmatize_endpoints():
    # 1. Endpoint /pos-tag (Bahasa Indonesia)
    resp_pos = client.post("/pos-tag", json={
        "text": "Tumbuhan melakukan fotosintesis.",
        "refine_foreign": False
    })
    assert resp_pos.status_code == 200
    data_pos = resp_pos.json()
    assert "tags" in data_pos
    assert len(data_pos["tags"]) > 0
    tag_map = {item["token"]: item["tag"] for item in data_pos["tags"]}
    assert tag_map.get("Tumbuhan") == "NN"

    # 2. Endpoint /pos-tag dengan refine_foreign (Campuran)
    resp_mixed = client.post("/pos-tag", json={
        "text": "Mereka sedang running testing otomatis.",
        "refine_foreign": True
    })
    assert resp_mixed.status_code == 200
    mixed_tag_map = {item["token"]: item["tag"] for item in resp_mixed.json()["tags"]}
    assert "FW (VERB)" in mixed_tag_map.get("running", "")

    # 3. Endpoint /lemmatize
    resp_lem = client.post("/lemmatize", json={
        "text": "Siswa menuliskan jawaban sambil running program."
    })
    assert resp_lem.status_code == 200
    data_lem = resp_lem.json()
    assert "lemmatized_text" in data_lem
    # 'menuliskan' -> 'tulis', 'running' -> 'run'
    assert "tulis" in data_lem["lemmatized_text"]
    assert "run" in data_lem["lemmatized_text"]


# ---------------------------------------------------------------------------
# Test 21: Integrasi Koreksi Esai dengan Lemmatisasi Morfologi Fleksibel
# ---------------------------------------------------------------------------
def test_grade_essay_with_bilingual_lemmatization():
    # Kunci jawaban menggunakan bentuk dasar/infleksi berbeda
    payload = {
        "student_answer": "Green plants are producing foods by absorbing sunlight and carbon dioxide.",
        "reference_answers": [
            "Plants produce food by absorbing sunlight and carbon dioxide."
        ],
        "keywords": [
            {"keyword": "produce food", "weight": 2.0},
            {"keyword": "sunlight", "weight": 1.5},
            {"keyword": "carbon dioxide", "weight": 2.0}
        ],
        "rule_weight": 0.5,
        "lsa_weight": 0.5,
    }
    r = client.post("/grade", json=payload).json()
    # 'producing foods' di-lemmatize menjadi 'produce food' sehingga keyword match 100%
    assert r["rule_based_score"] == 100.0
    assert r["lsa_score"] > 85.0
    assert "produce food (matched: produce food)" in r["matched_keywords"]


# ---------------------------------------------------------------------------
# Test 22: Feature Flags Preprocessing (Unit Test)
# ---------------------------------------------------------------------------
def test_preprocessing_feature_flags_unit():
    from nlp_pipeline.preprocessing import preprocess_text

    text = "<p>Green Plants are <b>producing</b> 5 foods in 1945, don't stop!</p>"

    # 1. Default (Semua flags aktif)
    default_out = preprocess_text(text)
    assert "<p>" not in default_out
    assert "five" in default_out  # num2words aktif
    assert "1945" in default_out  # tahun dipreservasi
    assert "produce" in default_out  # lemmatisasi aktif
    assert "dont" not in default_out and "not" in default_out  # kontraksi diekspansi

    # 2. Matikan Lemmatisasi (use_lemmatization=False)
    no_lem_out = preprocess_text(text, use_lemmatization=False)
    assert "producing" in no_lem_out  # tetap 'producing', tidak di-lemmatize jadi 'produce'

    # 3. Matikan Stopword Removal (remove_stopwords=False)
    no_stop_out = preprocess_text(text, remove_stopwords=False)
    # Stopword bahasa Inggris seperti 'in' atau 'are' tetap ada di output
    assert "in" in no_stop_out.split() or "are" in no_stop_out.split()

    # 4. Matikan Preservasi Tahun (preserve_years=False)
    no_year_out = preprocess_text(text, preserve_years=False)
    assert "1945" not in no_year_out

    # 5. Matikan Case Folding (case_folding=False)
    no_case_out = preprocess_text("Tumbuhan Hijau", case_folding=False, use_lemmatization=False)
    assert "Tumbuhan" in no_case_out and "Hijau" in no_case_out


# ---------------------------------------------------------------------------
# Test 23: Feature Flags On/Off pada API Grading
# ---------------------------------------------------------------------------
def test_preprocessing_flags_in_api_grading():
    payload_base = {
        "student_answer": "Tumbuhan melakukan proses fotosintesis.",
        "reference_answers": ["Tumbuhan melakukan proses fotosintesis."],
        "keywords": [{"keyword": "tumbuh", "weight": 1.0}],  # Kunci dalam bentuk kata dasar
        "rule_weight": 1.0,
        "lsa_weight": 0.0,
    }

    # Kasus A: Lemmatisasi AKTIF (default)
    # Siswa menulis 'tumbuhan' -> di-lemmatize jadi 'tumbuh' -> cocok dengan keyword 'tumbuh'
    res_with_lem = client.post("/grade", json={**payload_base, "use_lemmatization": True}).json()
    assert res_with_lem["rule_based_score"] == 100.0

    # Kasus B: Lemmatisasi DINONAKTIFKAN (use_lemmatization=False)
    # Siswa menulis 'tumbuhan' -> tetap 'tumbuhan' -> tidak cocok dengan keyword 'tumbuh'
    res_without_lem = client.post("/grade", json={**payload_base, "use_lemmatization": False}).json()
    assert res_without_lem["rule_based_score"] == 0.0

    # Kasus C: Mengirim konfigurasi via objek nested preprocessing_flags
    res_nested = client.post("/grade", json={
        **payload_base,
        "preprocessing_flags": {
            "use_lemmatization": False
        }
    }).json()
    assert res_nested["rule_based_score"] == 0.0


# ---------------------------------------------------------------------------
# Test 24: Ekspansi Sinonim Otomatis pada Rule-Based (ID & EN + Toggle Flag)
# ---------------------------------------------------------------------------
def test_synonym_expansion_rule_based():
    # Guru hanya menulis keyword tunggal "mencemari" dan "konsumtif" tanpa garis miring '/'
    # Siswa menulis sinonimnya: "mengotori" dan "boros"
    payload = {
        "student_answer": "Limbah pabrik mengotori sungai dan perilaku boros merugikan masyarakat.",
        "reference_answers": [
            "Limbah pabrik mencemari sungai dan perilaku konsumtif merugikan masyarakat."
        ],
        "keywords": [
            {"keyword": "mencemari", "weight": 2.0},
            {"keyword": "konsumtif", "weight": 2.0},
        ],
        "rule_weight": 0.7,
        "lsa_weight": 0.3,
    }

    # 1. Default (use_synonyms=True): kedua keyword harus cocok via ekspansi sinonim otomatis
    res_syn_on = client.post("/grade", json=payload).json()
    assert res_syn_on["rule_based_score"] == 100.0
    assert len(res_syn_on["matched_keywords"]) == 2

    # 2. Jika use_synonyms=False: tidak boleh mencocokkan sinonim yang tidak ditulis eksplisit
    res_syn_off = client.post("/grade", json={**payload, "use_synonyms": False}).json()
    assert res_syn_off["rule_based_score"] == 0.0
    assert len(res_syn_off["matched_keywords"]) == 0


# ---------------------------------------------------------------------------
# Test 25: Keamanan Strict Mode (**) dan Polysemy Guard pada Ekspansi Sinonim
# ---------------------------------------------------------------------------
def test_synonym_expansion_strict_and_polysemy_guard():
    # Keyword strict=True TIDAK BOLEH diekspansi ke sinonim
    payload_strict = {
        "student_answer": "Gaya hidup boros harus dihindari.",
        "reference_answers": ["Gaya hidup konsumtif harus dihindari."],
        "keywords": [
            {"keyword": "konsumtif", "weight": 1.0, "strict": True},
        ],
        "rule_weight": 1.0,
        "lsa_weight": 0.0,
    }
    res_strict = client.post("/grade", json=payload_strict).json()
    assert res_strict["rule_based_score"] == 0.0

    # Kata ambigu/polisemi seperti "bunga" atau "bisa" tidak boleh diekspansi sembarangan
    from nlp_pipeline.synonym_expander import get_synonyms
    assert len(get_synonyms("bisa")) == 0
    assert len(get_synonyms("bunga")) == 0


# ---------------------------------------------------------------------------
# Test 26: Penyelarasan Token Kanonik (Canonical Token Alignment) pada LSA
# ---------------------------------------------------------------------------
def test_lsa_canonical_token_alignment():
    # Siswa menggunakan kosakata sinonim penuh dari referensi guru
    payload = {
        "student_answer": "Binatang dan tanaman di rimba wajib dilindungi agar ekosistem tetap lestari.",
        "reference_answers": [
            "Hewan dan tumbuhan di hutan harus dijaga agar lingkungan tetap lestari."
        ],
        "keywords": [],
        "rule_weight": 0.0,
        "lsa_weight": 1.0,
    }

    res_aligned = client.post("/grade", json={**payload, "use_synonyms": True}).json()
    res_unaligned = client.post("/grade", json={**payload, "use_synonyms": False}).json()

    # Dengan Canonical Token Alignment, skor LSA harus jauh lebih tinggi dibanding tanpa alignment
    assert res_aligned["lsa_score"] > res_unaligned["lsa_score"]
    assert res_aligned["lsa_score"] >= 70.0


# ---------------------------------------------------------------------------
# Test 27: Keamanan Penalti Negasi pada LSA Canonical Token Alignment
# ---------------------------------------------------------------------------
def test_lsa_synonym_alignment_preserves_negation_penalty():
    ref = ["Sampah plastik mencemari lingkungan laut dan merusak ekosistem."]

    # Siswa A: positif menggunakan sinonim ("mengotori", "menghancurkan")
    res_pos = client.post("/grade", json={
        "student_answer": "Limbah plastik mengotori alam bahari dan menghancurkan ekosistem.",
        "reference_answers": ref,
        "keywords": [],
        "rule_weight": 0.0,
        "lsa_weight": 1.0,
    }).json()

    # Siswa B: kalimat negasi menggunakan sinonim ("tidak mengotori", "tidak menghancurkan")
    res_neg = client.post("/grade", json={
        "student_answer": "Limbah plastik tidak mengotori alam bahari dan tidak menghancurkan ekosistem.",
        "reference_answers": ref,
        "keywords": [],
        "rule_weight": 0.0,
        "lsa_weight": 1.0,
    }).json()

    # Skor kalimat bernegasi harus tetap dipotong oleh negation penalty
    assert res_neg["lsa_score"] < res_pos["lsa_score"]


# ---------------------------------------------------------------------------
# Test 28: Pelindung Kata Dasar Bahasa Indonesia Berakhiran '-s' (Language-Aware)
# ---------------------------------------------------------------------------
def test_language_aware_lemmatization_preserves_id_root_words():
    from nlp_pipeline.preprocessing import preprocess_text, _normalize_word

    # Kata dasar Bahasa Indonesia berakhiran '-s' tidak boleh dipotong oleh spaCy EN
    for word in ["boros", "kardus", "pedas", "gelas", "panas"]:
        assert _normalize_word(word) == word
        assert preprocess_text(word) == word

    # Dalam kalimat Bahasa Indonesia penuh
    assert "boros" in preprocess_text("Perilaku boros dan membeli kardus itu sangat merugikan.").split()
    # Dalam kalimat campuran (code-mixing), kata EN tetap terlemmatize
    assert "run" in preprocess_text("Siswa sedang running aplikasi baru.").split()


# ---------------------------------------------------------------------------
# Test 29: Perbaikan Strict Mode (Boundary Tanda Baca Ujung & HTML/Kutip Lengkung)
# ---------------------------------------------------------------------------
def test_strict_punctuation_boundaries_and_html_quotes():
    # 1. Kendala 1: Tanda baca di akhir/awal varian harus tetap menjaga batas kata di sisi satunya
    payload_boundary = {
        "reference_answers": ["Ada sungai, dan, hutan."],
        "keywords": [
            {"keyword": "dan,", "weight": 1.0, "strict": True},
            {"keyword": '"Halo', "weight": 1.0, "strict": True},
        ],
        "rule_weight": 1.0,
        "lsa_weight": 0.0,
    }
    # "badan," dan '"Halogen' TIDAK BOLEH false positive!
    r_false = client.post("/grade", json={
        **payload_boundary,
        "student_answer": 'Dia berkata "Halogen tidak bagus untuk badan, ya.'
    }).json()
    assert r_false["rule_based_score"] == 0.0

    # Jawaban presisi harus cocok keduanya
    r_true = client.post("/grade", json={
        **payload_boundary,
        "student_answer": 'Dia berkata "Halo di sungai, dan, hutan.'
    }).json()
    assert r_true["rule_based_score"] == 100.0

    # 2. Kendala 2: HTML tag terputus & kutip tipografi lengkung (“ ”) harus tetap cocok
    payload_html_quote = {
        "reference_answers": ['Ibu kota adalah Jakarta, katanya "Halo,"'],
        "keywords": [
            {"keyword": "Jakarta,", "weight": 1.0, "strict": True},
            {"keyword": '"Halo,"', "weight": 1.0, "strict": True},
        ],
        "rule_weight": 1.0,
        "lsa_weight": 0.0,
    }
    r_html_quote = client.post("/grade", json={
        **payload_html_quote,
        "student_answer": "<p>Ibu kota adalah <b>Jakarta</b>, katanya “Halo,” pelan.</p>"
    }).json()
    assert r_html_quote["rule_based_score"] == 100.0


# ---------------------------------------------------------------------------
# Test 30: Strict Per-Varian ('**' hanya mengikat 1 varian di kanannya di dalam '/')
# ---------------------------------------------------------------------------
def test_strict_per_variant_slash_isolation():
    from evaluate_model import parse_keywords

    # Pilar campuran: **Jakarta strict, tetapi "ibukota" di pilar yang sama tetap non-strict
    parsed = parse_keywords("**Jakarta / ibukota : 2.0; **Pancasila / dasar negara : 1.0")
    assert len(parsed) == 2
    assert parsed[0].strict is False  # Per-varian ditangani oleh rule_based.py

    # 1. Siswa menulis "jakarta" (huruf kecil) dan "dasar negara" (huruf kecil) ->
    #    **Jakarta GAGAL, ibukota tidak ada; **Pancasila gagal, tetapi "dasar negara" (non-strict) COCOK!
    res_partial = client.post("/grade", json={
        "student_answer": "Ibu kota negara itu jakarta dengan dasar negara yang kuat.",
        "reference_answers": ["Ibu kota negara adalah Jakarta dengan Pancasila sebagai dasar negara."],
        "keywords": [k.model_dump() for k in parsed],
        "rule_weight": 1.0,
        "lsa_weight": 0.0,
    }).json()
    assert res_partial["rule_based_score"] == 33.33  # hanya bobot 1.0 dari total 3.0

    # 2. Siswa menulis "IBUKOTA" (non-strict di pilar yang diawali **Jakarta!) dan "Pancasila" -> Keduanya COCOK!
    res_full = client.post("/grade", json={
        "student_answer": "IBUKOTA negara itu menggunakan Pancasila.",
        "reference_answers": ["Ibu kota negara adalah Jakarta dengan Pancasila."],
        "keywords": [k.model_dump() for k in parsed],
        "rule_weight": 1.0,
        "lsa_weight": 0.0,
    }).json()
    assert res_full["rule_based_score"] == 100.0


# ---------------------------------------------------------------------------
# Test 31: Proteksi Stopword pada Kata Kunci Non-Strict & Kurasi SEMANTIC_GUARD_WORDS
# ---------------------------------------------------------------------------
def test_non_strict_keyword_stopword_protection():
    # 1. "deskripsi umum" tanpa '**' tidak boleh terpotong jadi "deskripsi" saja
    payload_umum = {
        "reference_answers": ["Paragraf pertama berisi deskripsi umum."],
        "keywords": [
            {"keyword": "deskripsi umum", "weight": 1.0, "strict": False},
        ],
        "rule_weight": 1.0,
        "lsa_weight": 0.0,
    }
    # Siswa menulis berbagai variasi kapitalisasi -> harus selalu COCOK
    for variant_ans in ["DESKRIPSI UMUM", "Deskripsi Umum", "deskripsi umum", "DeSkRiPsI uMuM"]:
        res_ok = client.post("/grade", json={**payload_umum, "student_answer": variant_ans}).json()
        assert res_ok["rule_based_score"] == 100.0, f"Gagal pada variasi: {variant_ans}"

    # Siswa menulis "deskripsi bagian" -> TIDAK BOLEH cocok dengan "deskripsi umum"!
    res_diff = client.post("/grade", json={
        **payload_umum,
        "student_answer": "Paragraf kedua berisi deskripsi bagian."
    }).json()
    assert res_diff["rule_based_score"] == 0.0

    # 2. Frasa kunci yang mengandung stopword murni ("di atas") tetap terlindungi
    payload_prep = {
        "reference_answers": ["Buku berada di atas meja."],
        "keywords": [
            {"keyword": "di atas", "weight": 1.0, "strict": False},
        ],
        "rule_weight": 1.0,
        "lsa_weight": 0.0,
    }
    res_prep_ok = client.post("/grade", json={
        **payload_prep,
        "student_answer": "Buku itu terletak DI ATAS meja."
    }).json()
    assert res_prep_ok["rule_based_score"] == 100.0

    res_prep_fail = client.post("/grade", json={
        **payload_prep,
        "student_answer": "Buku itu terletak di bawah meja."
    }).json()
    assert res_prep_fail["rule_based_score"] == 0.0


# ---------------------------------------------------------------------------
# Test 32: Pra-Ekstraksi Koordinat Matematis (Q, P) dan (P, Q) pada SymPy
# ---------------------------------------------------------------------------
def test_math_coordinate_pairs_and_reversed_order():
    base_3c = {
        "reference_answers": ["Eq (Q:P): Eq (15:30000)"],
        "keywords": [],
        "math_steps": [
            {"equation": "P = 30000", "weight": 2.5},
            {"equation": "Q = 15", "weight": 2.5},
        ],
        "rule_weight": 0.0,
        "lsa_weight": 0.0,
        "math_weight": 1.0,
    }

    # 1. Kasus Elsinta: `eq (15 ; 30.000)` -> konvensi (Q, P)
    r_elsinta = client.post("/grade", json={**base_3c, "student_answer": "eq (15 ; 30.000)"}).json()
    assert r_elsinta["math_score"] == 100.0

    # 2. Koordinat standalone tanpa awalan `eq`: `(15, 30.000)`
    r_standalone = client.post("/grade", json={**base_3c, "student_answer": "(15, 30.000)"}).json()
    assert r_standalone["math_score"] == 100.0

    # 3. Kasus Achmad dengan label sumbu eksplisit `(P, Q) = (30.000, 15)`
    r_pq_labeled = client.post("/grade", json={
        **base_3c,
        "student_answer": "c. (P, Q) = (30.000, 15)"
    }).json()
    assert r_pq_labeled["math_score"] == 100.0

    # 4. Kasus Nadira tanpa label `(P, Q)`: `(c) Eq (30.000 ; 15)` -> terbalik, tidak boleh lolos
    r_reversed = client.post("/grade", json={
        **base_3c,
        "student_answer": "(c) Eq (30.000 ; 15)"
    }).json()
    assert r_reversed["math_score"] == 0.0

    # 5. Kasus desimal koma Indonesia Soal 4a: `Titik Keseimbangan E (7,5 ; 4,5)`
    r_4a = client.post("/grade", json={
        "student_answer": "Titik Keseimbangan E (7,5 ; 4,5)",
        "reference_answers": ["Titik keseimbangan : E = (Q=7,5 ; P=4,5)"],
        "keywords": [],
        "math_steps": [
            {"equation": "Q = 7.5", "weight": 5.0},
            {"equation": "P = 4.5", "weight": 5.0},
        ],
        "rule_weight": 0.0,
        "lsa_weight": 0.0,
        "math_weight": 1.0,
    }).json()
    assert r_4a["math_score"] == 100.0


# ---------------------------------------------------------------------------
# Test 33: Ekstraksi Frasa Verbal Code-Switching & Auto-Discovery Label Variabel
# ---------------------------------------------------------------------------
def test_math_verbal_codeswitching_and_dynamic_labels():
    base_3c = {
        "reference_answers": ["P = 30000, Q = 15"],
        "keywords": [],
        "math_steps": [
            {"equation": "P = 30000", "weight": 2.5},
            {"equation": "Q = 15", "weight": 2.5},
        ],
        "rule_weight": 0.0,
        "lsa_weight": 0.0,
        "math_weight": 1.0,
    }

    # 1. Kasus Khalefa: `c) equilibrium is at the price Rp. 30.000 with quantity of 15.`
    r_khalefa = client.post("/grade", json={
        **base_3c,
        "student_answer": "c) equilibrium is at the price Rp. 30.000 with quantity of 15."
    }).json()
    assert r_khalefa["math_score"] == 100.0

    # 2. Frasa verbal Bahasa Indonesia + unit
    r_id_verbal = client.post("/grade", json={
        **base_3c,
        "student_answer": "Jadi harga sebesar Rp 30.000 dengan kuantitas 15 unit."
    }).json()
    assert r_id_verbal["math_score"] == 100.0

    # 3. Auto-discovery lintas mata pelajaran (Fisika: v & t dari kunci jawaban guru)
    r_physics = client.post("/grade", json={
        "student_answer": "Hasil perhitungannya adalah kelajuan 20 dengan durasi 5.",
        "reference_answers": ["Diketahui Kelajuan (v) adalah 20 dan Durasi (t) adalah 5."],
        "keywords": [],
        "math_steps": [
            {"equation": "v = 20", "weight": 1.0},
            {"equation": "t = 5", "weight": 1.0},
        ],
        "rule_weight": 0.0,
        "lsa_weight": 0.0,
        "math_weight": 1.0,
    }).json()
    assert r_physics["math_score"] == 100.0


# ---------------------------------------------------------------------------
# Test 34: Persamaan Berantai (=), Mata Uang Rp, dan Alias Pe/Qe pada SymPy
# ---------------------------------------------------------------------------
def test_math_currency_chained_eq_and_equilibrium_aliases():
    base_3c = {
        "reference_answers": ["P = 30000, Q = 15"],
        "keywords": [],
        "math_steps": [
            {"equation": "P = 30000", "weight": 2.5},
            {"equation": "Q = 15", "weight": 2.5},
        ],
        "rule_weight": 0.0,
        "lsa_weight": 0.0,
        "math_weight": 1.0,
    }

    # 1. Dua penugasan variabel dalam satu baris + alias Pe/Qe (Kasus Desta)
    r_desta = client.post("/grade", json={
        **base_3c,
        "student_answer": "c) Pe = 30.000 with the Qe = 15"
    }).json()
    assert r_desta["math_score"] == 100.0

    # 2. Satuan Rp sebelum angka + persamaan berantai `Q = -60.000 / -4.000 = 15`
    r_chained = client.post("/grade", json={
        **base_3c,
        "student_answer": "c. P = Rp 30.000\nQ = -60.000 / -4.000 = 15"
    }).json()
    assert r_chained["math_score"] == 100.0


# ---------------------------------------------------------------------------
# Test 35: Isolasi Garis Miring ('/') & Auto Switch Flag remove_punctuation=False
# ---------------------------------------------------------------------------
def test_math_slash_isolation_and_auto_punctuation_switch():
    from schemas import EssayRequest
    from evaluate_model import parse_math_steps, parse_keywords

    # 1. Parser memisahkan math_steps vs keywords tanpa merusak pembagian '/'
    steps = parse_math_steps("(P - 12000)/4000 = (Q - 4000)/-2000 : 2; Qd = 10000 - 0.5*P : 18")
    assert len(steps) == 2
    assert steps[0].equation == "(P - 12000)/4000 = (Q - 4000)/-2000"
    assert steps[0].weight == 2.0

    kws = parse_keywords("kewajiban / tanggung jawab : 2.0")
    assert len(kws) == 1 and kws[0].keyword == "kewajiban / tanggung jawab"

    # 2. Auto switch flag remove_punctuation=False pada soal berkategori hitungan
    math_req = EssayRequest(
        student_answer="(P - 12.000)/4.000 = (Q - 4.000)/-2.000\nQd = 10.000 - 0,5P",
        reference_answers=["(P - 12000)/4000 = (Q - 4000)/-2000"],
        math_steps=steps,
        math_weight=1.0,
        rule_weight=0.0,
        lsa_weight=0.0,
    )
    assert math_req.get_preprocessing_flags().remove_punctuation is False

    # Tetapi jika eksplisit di-override user ke True, tetap dihormati
    math_req_override = EssayRequest(
        student_answer="Qd = 10000 - 0.5P",
        reference_answers=["Qd = 10000 - 0.5P"],
        math_steps=steps,
        math_weight=1.0,
        remove_punctuation=True,
    )
    assert math_req_override.get_preprocessing_flags().remove_punctuation is True

    # Evaluasi API penuh dengan pembagian '/' menghasilkan math_score = 100.0
    res_api = client.post("/grade", json={
        "student_answer": "(P - 12.000)/4.000 = (Q - 4.000)/-2.000\nQd = 10.000 - 0,5P",
        "reference_answers": ["(P - 12000)/4000 = (Q - 4000)/-2000"],
        "math_steps": [s.model_dump() for s in steps],
        "math_weight": 1.0,
        "rule_weight": 0.0,
        "lsa_weight": 0.0,
    }).json()
    assert res_api["math_score"] == 100.0


# ---------------------------------------------------------------------------
# Test 36: Auto-Trigger Mode (linguistic, exact, conceptual) dari Subject CBT
# ---------------------------------------------------------------------------
def test_subject_and_essay_mode_auto_triggers():
    from schemas import EssayRequest, MathStepItem

    # 1. Trigger Linguistic dari format dropdown CBT ("Bahasa Indonesia (XI)", "Bahasa Inggris (X)")
    req_bindo = EssayRequest(
        subject="Bahasa Indonesia (XI)",
        student_answer="Tumbuhan melakukan fotosintesis",
        reference_answers=["Tumbuhan melakukan fotosintesis"],
    )
    assert req_bindo.resolve_essay_mode() == "linguistic"
    flags_bindo = req_bindo.get_preprocessing_flags()
    assert flags_bindo.use_lemmatization is True
    assert flags_bindo.use_synonyms is True
    assert flags_bindo.convert_numbers is True
    assert flags_bindo.remove_punctuation is True

    # 2. Trigger Exact dari nama mapel ("Matematika Peminatan (XII)", "Fisika (X)")
    req_math_subj = EssayRequest(
        subject="Matematika Peminatan (XII)",
        student_answer="Q = 100 - 2P",
        reference_answers=["Q = 100 - 2P"],
    )
    assert req_math_subj.resolve_essay_mode() == "exact"
    flags_math = req_math_subj.get_preprocessing_flags()
    assert flags_math.remove_punctuation is False
    assert flags_math.convert_numbers is False
    assert flags_math.use_lemmatization is False
    assert flags_math.use_synonyms is False

    # 3. Trigger Conceptual dari nama mapel ("Informatika (XI)", "Sejarah Indonesia (X)")
    req_info = EssayRequest(
        subject="Informatika (XI)",
        student_answer="Algoritma adalah urutan langkah logis",
        reference_answers=["Algoritma merupakan langkah penyelesaian masalah"],
    )
    assert req_info.resolve_essay_mode() == "conceptual"
    flags_info = req_info.get_preprocessing_flags()
    assert flags_info.use_lemmatization is True
    assert flags_info.use_synonyms is True
    assert flags_info.preserve_years is True

    # 4. Smart Hybrid Trigger pada Ekonomi:
    #    - Tanpa math_steps -> conceptual
    #    - Dengan math_steps -> otomatis berubah menjadi exact
    req_econ_theory = EssayRequest(
        subject="Ekonomi (XI)",
        student_answer="Inflasi adalah kenaikan harga barang secara umum",
        reference_answers=["Inflasi adalah kenaikan harga secara umum dan terus-menerus"],
    )
    assert req_econ_theory.resolve_essay_mode() == "conceptual"

    req_econ_math = EssayRequest(
        subject="Ekonomi (XI)",
        student_answer="Q = 15 dan P = 30000",
        reference_answers=["Q = 15, P = 30000"],
        math_steps=[MathStepItem(equation="Q = 15", weight=1.0)],
        math_weight=1.0,
    )
    assert req_econ_math.resolve_essay_mode() == "exact"

    # 5. Verifikasi response endpoint /grade menyertakan essay_mode yang aktif
    res_api = client.post("/grade", json={
        "subject": "Bahasa Inggris (X)",
        "student_answer": "Plants produce food using sunlight.",
        "reference_answers": ["Plants make food using sunlight."],
        "keywords": [{"keyword": "produce", "weight": 1.0}],
    }).json()
    assert res_api["essay_mode"] == "linguistic"


