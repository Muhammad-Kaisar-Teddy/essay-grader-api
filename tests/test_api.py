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
