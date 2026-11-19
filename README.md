# Hybrid Automated Essay Grading (AEG) API

Engine dan REST API Penilaian Esai Otomatis (*Automated Essay Grading*) menggunakan arsitektur **Hybrid**:
1. **Rule-Based Keyword Matching** dengan pembobotan fleksibel, pencocokan ketat (*strict mode*), dan penanganan negasi.
2. **Evaluasi Langkah Matematis (SymPy)** untuk mata pelajaran eksakta/ekonomi yang memverifikasi kesetaraan aljabar dan toleransi numerik.
3. **Latent Semantic Analysis (LSA)** berbasis TF-IDF, TruncatedSVD, dan Cosine Similarity untuk mengukur kedekatan makna konseptual terhadap kunci jawaban.
4. **Bilingual NLP Pipeline** (Bahasa Indonesia & Bahasa Inggris) terintegrasi menggunakan *nlp-id*, *Sastrawi*, *spaCy*, *WordNet*, ekspansi sinonim otomatis, serta penanganan bilangan (*num2words*).

---

## Daftar Isi
- [1. Prasyarat Sistem](#1-prasyarat-sistem)
- [2. Panduan Instalasi & Setup Lingkungan](#2-panduan-instalasi--setup-lingkungan)
- [3. Cara Menjalankan & Menggunakan](#3-cara-menjalankan--menggunakan)
  - [A. Menjalankan REST API Server (FastAPI)](#a-menjalankan-rest-api-server-fastapi)
  - [B. Uji Coba Interaktif di Terminal (`interactive_test.py`)](#b-uji-coba-interaktif-di-terminal-interactive_testpy)
  - [C. Menjalankan Unit Test Otomatis (`pytest`)](#c-menjalankan-unit-test-otomatis-pytest)
  - [D. Menjalankan via Docker](#d-menjalankan-via-docker)
- [4. Panduan Lengkap Evaluasi Akurasi Model (`evaluate_model.py`)](#4-panduan-lengkap-evaluasi-akurasi-model-evaluate_modelpy)
  - [A. Sintaks Dasar](#a-sintaks-dasar)
  - [B. Parameter & Argumen CLI](#b-parameter--argumen-cli)
  - [C. Contoh Skenario Penggunaan Terminal](#c-contoh-skenario-penggunaan-terminal)
  - [D. Panduan Membaca Metrik Evaluasi](#d-panduan-membaca-metrik-evaluasi)
  - [E. Lokasi Penyimpanan Hasil Evaluasi](#e-lokasi-penyimpanan-hasil-evaluasi)
- [5. Struktur Direktori Proyek](#5-struktur-direktori-proyek)
- [6. Troubleshooting & Catatan Penting](#6-troubleshooting--catatan-penting)

---

## 1. Prasyarat Sistem

Sebelum memulai instalasi di komputer/perangkat lokal Anda, pastikan telah terpasang:
- **Git** ([Unduh Git](https://git-scm.com/))
- **Python 3.10 – 3.12** ([Unduh Python](https://www.python.org/downloads/)) *(direkomendasikan Python 3.11 atau 3.12)*
- *(Opsional)* **Docker** jika ingin menjalankan aplikasi via kontainer tanpa konfigurasi Python lokal.

---

## 2. Panduan Instalasi & Setup Lingkungan

Ikuti langkah-langkah berikut untuk mengimplementasikan dan menjalankan repository ini dari awal:

### Langkah 1: Clone Repository
Buka terminal (PowerShell, Command Prompt, atau Terminal Linux/macOS):
```bash
git clone <URL_REPOSITORY_ANDA>
cd essay-grader-api
```

### Langkah 2: Buat & Aktifkan Virtual Environment
Sangat disarankan membuat virtual environment agar dependensi tidak berbenturan dengan sistem global:

- **Di Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
  > *Catatan Windows PowerShell:* Jika muncul pesan peringatan *Execution Policy*, jalankan perintah berikut sekali:
  > ```powershell
  > Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  > .\.venv\Scripts\Activate.ps1
  > ```

- **Di Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### Langkah 3: Instal Dependensi
Perbarui `pip` lalu pasang seluruh dependensi yang tertera di `requirements.txt`:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```
*(Paket mencakup FastAPI, Uvicorn, Scikit-Learn, SymPy, Sastrawi, nlp-id, spaCy, dan model `en_core_web_sm`).*

### Langkah 4: Unduh Korpus & Model Bahasa Tambahan (*One-time Setup*)
Jalankan satu baris perintah berikut untuk mengunduh korpus NLTK dan menginisialisasi model `nlp-id`:
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); from nlp_id import PosTag; PosTag()"
```

---

## 3. Cara Menjalankan & Menggunakan

### A. Menjalankan REST API Server (FastAPI)
Untuk menjalankan server API lokal untuk integrasi web, mobile, atau LMS:

```bash
uvicorn main:app --reload --port 8000
```

- **Akses API:** `http://localhost:8000`
- **Dokumentasi Interaktif (Swagger UI):** Buka browser ke [`http://localhost:8000/docs`](http://localhost:8000/docs)
- **Dokumentasi Alternatif (ReDoc):** [`http://localhost:8000/redoc`](http://localhost:8000/redoc)

#### Endpoint Utama:
| Method | Endpoint | Deskripsi |
| :--- | :--- | :--- |
| `POST` | `/grade` | Menilai satu jawaban esai siswa secara komprehensif. |
| `POST` | `/grade-batch` | Menilai daftar jawaban esai secara kolektif/masal. |
| `POST` | `/pos-tag` | Mendapatkan tag Part-of-Speech token kata (bilingual). |
| `POST` | `/lemmatize` | Menguji modul lemmatisasi teks secara langsung. |

#### Contoh Payload Pengujian `POST /grade`:
```json
{
  "subject": "Bahasa Indonesia",
  "student_answer": "Fotosintesis adalah proses tumbuhan hijau menghasilkan oksigen dan glukosa menggunakan energi cahaya matahari.",
  "reference_answers": [
    "Fotosintesis merupakan mekanisme tumbuhan memproduksi glukosa dan oksigen dengan bantuan sinar matahari serta klorofil."
  ],
  "keywords": [
    {"keyword": "fotosintesis", "weight": 1.0, "strict": false},
    {"keyword": "oksigen", "weight": 1.0, "strict": false},
    {"keyword": "glukosa", "weight": 1.0, "strict": false},
    {"keyword": "cahaya matahari", "weight": 1.0, "strict": false}
  ],
  "rule_weight": 0.5,
  "lsa_weight": 0.5,
  "math_weight": 0.0,
  "max_score": 100.0,
  "tgate": 0.0
}
```

---

### B. Uji Coba Interaktif di Terminal (`interactive_test.py`)
Jika ingin menguji alur grading dan melihat visualisasi setiap langkah secara langsung di terminal:

```bash
python interactive_test.py
```

Skrip ini akan memandu Anda:
1. Memasukkan konfigurasi bobot (Rule, Math, LSA).
2. Memilih mode esai (`linguistic`, `exact`, atau `conceptual`).
3. Menginput kunci jawaban, kata kunci, dan persamaan matematika.
4. Menampilkan demo preprocessing, deteksi bahasa, ekspansi sinonim, hingga rincian kalkulasi skor akhir.

---

### C. Menjalankan Unit Test Otomatis (`pytest`)
Untuk memvalidasi bahwa seluruh komponen NLP, parsing rumus matematika, dan endpoint API berfungsi dengan baik:

```bash
pytest
```
Atau untuk melihat log detail pengujian:
```bash
pytest -v -s
```

---

### D. Menjalankan via Docker
Jika tidak ingin melakukan instalasi Python secara lokal, Anda dapat menggunakan Docker:

1. **Build image Docker:**
   ```bash
   docker build -t essay-grader-api .
   ```
2. **Jalankan kontainer:**
   ```bash
   docker run -d -p 8000:10000 --name essay-grader essay-grader-api
   ```
3. Buka browser pada [`http://localhost:8000/docs`](http://localhost:8000/docs).

---

## 4. Panduan Lengkap Evaluasi Akurasi Model (`evaluate_model.py`)

Skrip [`evaluate_model.py`](evaluate_model.py) digunakan untuk menguji dan mengevaluasi akurasi penilaian model AI dibandingkan dengan penilaian riil guru (*ground truth*) pada dataset siswa di folder `test_set/`.

### A. Sintaks Dasar

```powershell
python evaluate_model.py [OPSI...]
```

> **Tips Windows PowerShell:** Selalu gunakan awalan `python evaluate_model.py` atau path interpreter venv `.\.venv\Scripts\python.exe evaluate_model.py` untuk menghindari error *CommandNotFoundException*.

---

### B. Parameter & Argumen CLI

#### 1. Pengaturan Data Uji & Mata Pelajaran
| Parameter | Tipe | Nilai Default | Deskripsi |
| :--- | :---: | :---: | :--- |
| `--subject` | `str` | `bahasa_indonesia` | Pilihan mata pelajaran: `bahasa_indonesia`, `english`, `economy`, `sejarah`, atau `all` (seluruh mapel sekaligus). |
| `--ref-file` | `str` | *File pertama yang cocok* | Nama/path file kunci jawaban spesifik (contoh: `reference_answer_bindo_final.xlsx`). |
| `--limit` | `int` | `None` (Semua) | Membatasi jumlah file siswa untuk uji cepat (contoh: `--limit 5`). |
| `--output-dir` | `str` | `test_set/test_result` | Direktori tempat menyimpan file laporan evaluasi (diabaikan oleh git). |
| `--tgate` | `float` | `0.0` | Threshold Gate (0 - 100). LSA hanya diaktifkan jika skor faktual rule-based $\ge$ tgate. |
| `--ignore-auto-status` | *flag* | `False` | Mengabaikan kolom `auto_grading_status` dan memaksa seluruh butir dinilai murni oleh AI. |
| `--mode` | `str` | `None` (Auto) | Memaksa mode esai tertentu (`linguistic`, `exact`, `conceptual`). |

#### 2. Feature Flags Preprocessing (Ablation Study)
Secara default, seluruh tahapan preprocessing aktif sesuai mode esai. Anda dapat mematikan fitur spesifik untuk analisis pengaruh modul:

| Flag Preprocessing | Efek Saat Ditambahkan |
| :--- | :--- |
| `--no-lemmatization` | **Mematikan lemmatisasi** (*nlp-id* untuk ID, *spaCy* untuk EN). Kata tidak diubah ke bentuk dasar. |
| `--no-stopwords` | **Mematikan stopword removal**. Kata-kata umum dan kata hubung tetap dipertahankan. |
| `--no-num2words` | **Mematikan konversi angka ke kata** (angka `10` tidak diubah menjadi `sepuluh`). |
| `--no-punctuation` | **Mematikan pembersihan tanda baca**. |
| `--no-casefolding` | **Mematikan case folding** (huruf kapital dipertahankan). |
| `--no-html` | **Mematikan sanitasi tag HTML** dan normalisasi baris baru. |
| `--no-years` | **Mematikan preservasi angka tahun** (rentang 1000–2099). |
| `--no-contractions` | **Mematikan ekspansi singkatan** bahasa Inggris (*don't* $\rightarrow$ *do not*). |
| `--no-synonyms` | **Mematikan ekspansi sinonim otomatis** (WordNet / Tesaurus Bilingual). |

#### 3. Flag Arsitektur & Suite Ablasi Otomatis
| Flag | Fungsi |
| :--- | :--- |
| `--force-rule-only` | Memaksa bobot Rule-based = 1.0 dan LSA = 0.0 (evaluasi murni kata kunci). |
| `--force-lsa-only` | Memaksa bobot LSA = 1.0 dan Rule-based = 0.0 (evaluasi murni semantik LSA). |
| `--force-no-lsa` | Menonaktifkan komponen LSA secara penuh. |
| `--run-ablation` | **Menjalankan suite ablasi lengkap secara otomatis** (baseline + variasi ablasi) serta menghasilkan tabel perbandingan $\Delta$MAE, $\Delta$RMSE, dan $\Delta r$. |

---

### C. Contoh Skenario Penggunaan Terminal

#### Skenario 1: Uji Cepat (Smoke Test 5 Siswa)
```powershell
python evaluate_model.py --subject bahasa_indonesia --ref-file reference_answer_bindo_final.xlsx --limit 5
```

#### Skenario 2: Evaluasi Penuh Satu Mata Pelajaran
```powershell
python evaluate_model.py --subject bahasa_indonesia --ref-file reference_answer_bindo_final.xlsx
```

#### Skenario 3: Evaluasi Seluruh Mata Pelajaran Sekaligus (`--subject all`)
```powershell
python evaluate_model.py --subject all
```
*Skrip akan mengevaluasi seluruh mata pelajaran yang tersedia dan mencetak tabel rekapitulasi cross-subject.*

#### Skenario 4: Evaluasi Mata Pelajaran Eksakta / Bahasa Lain
```powershell
# Ekonomi (mendukung verifikasi persamaan matematis SymPy)
python evaluate_model.py --subject economy --ref-file reference_answer_economy.xlsx

# Bahasa Inggris
python evaluate_model.py --subject english --ref-file reference_answer_english.xlsx

# Sejarah
python evaluate_model.py --subject sejarah --ref-file reference_answer_history.xlsx
```

#### Skenario 5: Menjalankan Suite Studi Ablasi Otomatis
```powershell
# Jalankan suite studi ablasi otomatis untuk Bahasa Indonesia
python evaluate_model.py --subject bahasa_indonesia --ref-file reference_answer_bindo_final.xlsx --run-ablation

# Jalankan suite studi ablasi untuk seluruh mata pelajaran
python evaluate_model.py --subject all --run-ablation
```

#### Skenario 6: Studi Ablasi Manual Fitur Preprocessing
```powershell
# Uji coba tanpa modul lemmatisasi
python evaluate_model.py --subject bahasa_indonesia --ref-file reference_answer_bindo_final.xlsx --no-lemmatization

# Uji coba evaluasi murni Rule-Based (tanpa LSA)
python evaluate_model.py --subject bahasa_indonesia --ref-file reference_answer_bindo_final.xlsx --force-rule-only
```

---

### D. Panduan Membaca Metrik Evaluasi

Setelah evaluasi selesai, terminal akan menampilkan metrik statistik performa berikut:

1. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |\text{Skor Guru}_i - \text{Skor AI}_i|$$
   - Rata-rata selisih absolut nilai AI terhadap nilai guru per butir soal.
   - *Interpretasi:* Semakin kecil mendekati 0, semakin akurat penilaian AI.

2. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (\text{Skor Guru}_i - \text{Skor AI}_i)^2}$$
   - Sensitif terhadap kesalahan besar (*outlier*). Jika RMSE mendekati MAE, sistem stabil dan tidak menghasilkan deviasi ekstrem.

3. **Akurasi Sempurna (Exact Match, Error = 0):**
   - Persentase jawaban di mana nilai AI **100% identik** dengan nilai guru ($|\text{Guru} - \text{AI}| = 0$).

4. **Toleransi Selisih $\le 0.5$ Poin:**
   - Persentase butir jawaban dengan simpangan sangat tipis ($\le 0.5$ poin).

5. **Toleransi Selisih $\le 1.0$ Poin (*Adjacent Agreement*):**
   - Persentase jawaban dalam deviasi wajar maksimal 1.0 poin. Dalam literatur AES (*Automated Essay Scoring*), nilai di atas 90% dikategorikan sangat andal.

6. **Korelasi Pearson ($r$) & Spearman ($\rho$):**
   - Mengukur derajat kelinearan ($r$) dan urutan ranking ($\rho$) antara penilaian AI dan guru pada rentang $-1.0$ s.d. $+1.0$.
   - Nilai $r \ge 0.80$ dikategorikan sebagai **Korelasi Sangat Kuat (*Very Strong Correlation*)**.

---

### E. Lokasi Penyimpanan Hasil Evaluasi

Seluruh output pengujian disimpan otomatis ke folder:
`test_set/test_result/`

1. **`evaluation_<subject>_detailed.xlsx`**:  
   Tabel lengkap per baris jawaban berisi:
   - Nama siswa & ID soal
   - Teks jawaban siswa
   - Nilai guru vs Nilai AI
   - Selisih absolut (*abs_error*) dan kuadrat (*squared_error*)
   - Status autograding (`on` / `off`) dan sumber penilaian (`ai_model` / `teacher_override`).

2. **`evaluation_<subject>_summary.txt`**:  
   Ringkasan teks metrik statistik (MAE, RMSE, Pearson $r$, Spearman $\rho$, toleransi skor).

3. **`ablation_study_<subject>_<timestamp>.xlsx` / `.csv`**:  
   Hasil komparasi komprehensif metrik baseline terhadap setiap varian ablasi saat menggunakan flag `--run-ablation`.

> **Catatan Git:** Direktori `test_set/test_result/` telah terdaftar di `.gitignore` sehingga tidak akan mengotori riwayat commit repository.

---

## 5. Struktur Direktori Proyek

```text
essay-grader-api/
├── nlp_pipeline/               # Pipeline pemrosesan bahasa alami & scoring
│   ├── preprocessing.py        # Tokenisasi, lemmatisasi, stopword, num2words
│   ├── rule_based.py           # Pencocokan kata kunci & penanganan negasi
│   ├── math_evaluator.py       # Ekstraksi & evaluasi formula via SymPy
│   ├── lsa_model.py            # TF-IDF vectorizer & TruncatedSVD LSA
│   └── synonym_expander.py     # Ekspansi sinonim bilingual (WordNet/Tesaurus)
├── test_set/                   # Data uji per mata pelajaran & kunci jawaban
│   ├── bahasa_indonesia/
│   ├── economy/
│   ├── english/
│   ├── sejarah/
│   └── test_result/            # Folder output evaluasi (diabaikan git)
├── tests/                      # Unit testing otomatis
│   └── test_api.py             # Pengujian endpoint & modul pipeline
├── Dockerfile                  # Konfigurasi container image berbasis Python 3.12 slim
├── grader.py                   # Orkesrator penilaian hybrid (Rule + Math + LSA)
├── schemas.py                  # Skema validasi Pydantic v2 (Request/Response)
├── main.py                     # Entry point REST API (FastAPI)
├── interactive_test.py         # Skrip pengujian interaktif bertahap di terminal
├── evaluate_model.py           # Skrip evaluasi akurasi masal & studi ablasi
├── requirements.txt            # Daftar pustaka dependensi Python
└── README.md                   # Dokumentasi panduan teknis
```

---

## 6. Troubleshooting & Catatan Penting

1. **Error PowerShell: *cannot be loaded because running scripts is disabled on this system*:**  
   Buka PowerShell dan jalankan:
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   ```
   Lalu ulangi pengaktifan environment `.\.venv\Scripts\Activate.ps1`.

2. **Perintah `python` tidak dikenali atau salah versi:**  
   Gunakan path interpreter virtual environment secara eksplisit:
   - Windows: `.\.venv\Scripts\python.exe <nama_skrip.py>`
   - Linux/Mac: `./.venv/bin/python <nama_skrip.py>`

3. **Unduhan Korpus Pertama Kali:**  
   Pastikan perangkat terhubung ke internet saat pertama kali menjalankan langkah instalasi korpus NLTK dan `nlp-id`. Setelah itu, seluruh pipeline NLP dapat bekerja secara offline (*tanpa koneksi internet*).

4. **Kebutuhan Memori (RAM):**  
   Model dirancang sangat ringan (~200MB - 400MB RAM saat berjalan), sehingga sangat aman dijalankan di laptop standar maupun server cloud *free tier* (seperti Render / Hugging Face Spaces).
