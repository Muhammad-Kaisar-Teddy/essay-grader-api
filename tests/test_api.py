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







