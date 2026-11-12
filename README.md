# Panduan Evaluasi Model Penilaian Esai Otomatis (AEG)

Dokumen ini menjelaskan cara menjalankan pengujian dan evaluasi akurasi model penilaian esai otomatis (*Automated Essay Grading*) menggunakan antarmuka terminal/CLI melalui skrip [`evaluate_model.py`](evaluate_model.py).

---

## 1. Persiapan Lingkungan (Virtual Environment)

Sebelum menjalankan pengujian di terminal (PowerShell atau Command Prompt), pastikan direktori kerja aktif berada di root project (`essay-grader-api`) dan virtual environment telah aktif:

### Mengaktifkan Virtual Environment di PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

> **Catatan Windows PowerShell:**  
> Selalu jalankan skrip Python dengan mengetikkan `python` di awal (contoh: `python evaluate_model.py ...`) atau menggunakan path interpreter langsung `.\.venv\Scripts\python.exe evaluate_model.py ...` agar tidak muncul pesan error *CommandNotFoundException*.

---

## 2. Sintaks Dasar Perintah

```powershell
python evaluate_model.py [OPSI...]
```

---

## 3. Daftar Parameter & Argumen CLI

### A. Pengaturan Data Uji & Mata Pelajaran

| Parameter | Tipe | Nilai Default | Deskripsi |
| :--- | :---: | :---: | :--- |
| `--subject` | `str` | `bahasa_indonesia` | Nama folder mata pelajaran di dalam `test_set/`. Pilihan: `bahasa_indonesia`, `economy`, `english`, `sejarah`. |
| `--ref-file` | `str` | *File pertama yang cocok* | Nama file kunci jawaban spesifik (misal: `reference_answer_bindo_final.xlsx` atau `reference_answer_bindo_awal.xlsx`). |
| `--limit` | `int` | `None` (Semua) | Membatasi jumlah file siswa untuk pengujian cepat (misal: `--limit 5` untuk 5 siswa pertama). |
| `--output-dir` | `str` | `test_set/test_result` | Direktori tempat menyimpan file laporan evaluasi (otomatis diabaikan oleh `.gitignore`). |
| `--ignore-auto-status` | *flag* | `False` | Mengabaikan kolom `auto_grading_status` dan memaksa seluruh butir soal dinilai murni oleh model AI. |

---

### B. Feature Flags: Uji Coba Preprocessing (Ablation Study)

Secara default, seluruh tahapan preprocessing aktif. Anda dapat menonaktifkan fitur tertentu untuk keperluan analisis ablasi atau eksperimen akurasi:

| Flag Preprocessing | Efek Saat Ditambahkan |
| :--- | :--- |
| `--no-lemmatization` | **Mematikan lemmatisasi** (*nlp-id* untuk Bahasa Indonesia, *spaCy* untuk Bahasa Inggris). Kata tidak dikembalikan ke bentuk dasar. |
| `--no-stopwords` | **Mematikan stopword removal**. Kata-kata umum dan kata hubung tetap dipertahankan. |
| `--no-num2words` | **Mematikan konversi angka ke kata** (angka `10` tidak diubah menjadi `sepuluh`). |
| `--no-punctuation` | **Mematikan pembersihan tanda baca**. |
| `--no-casefolding` | **Mematikan case folding** (karakter huruf besar/kecil tidak diubah menjadi lowercase). |
| `--no-html` | **Mematikan sanitasi tag HTML** dan normalisasi jeda baris. |
| `--no-years` | **Mematikan preservasi angka tahun** (rentang 1000–2099). |
| `--no-contractions` | **Mematikan ekspansi singkatan** bahasa Inggris (*don't* $\rightarrow$ *do not*). |

---

## 4. Contoh Penggunaan di Terminal

### Skenario 1: Uji Cepat (Smoke Test 5 Siswa)
Mengevaluasi 5 siswa pertama pada mata pelajaran Bahasa Indonesia menggunakan kunci jawaban final:
```powershell
python evaluate_model.py --subject bahasa_indonesia --ref-file reference_answer_bindo_final.xlsx --limit 5
```

### Skenario 2: Evaluasi Penuh Seluruh Siswa
Mengevaluasi seluruh 119 siswa Bahasa Indonesia:
```powershell
python evaluate_model.py --subject bahasa_indonesia --ref-file reference_answer_bindo_final.xlsx
```

### Skenario 3: Membandingkan Kunci Jawaban Awal vs Final
Untuk menguji performa kunci jawaban versi awal:
```powershell
python evaluate_model.py --subject bahasa_indonesia --ref-file reference_answer_bindo_awal.xlsx
```

### Skenario 4: Studi Ablasi (Ablation Study)
Menguji seberapa besar kontribusi modul Lemmatisasi atau Stopword Removal terhadap akurasi:
```powershell
# Uji tanpa lemmatisasi
python evaluate_model.py --subject bahasa_indonesia --ref-file reference_answer_bindo_final.xlsx --no-lemmatization

# Uji tanpa stopword removal dan tanpa konversi angka
python evaluate_model.py --subject bahasa_indonesia --ref-file reference_answer_bindo_final.xlsx --no-stopwords --no-num2words
```

### Skenario 5: Menguji Mata Pelajaran Lain
```powershell
# Ekonomi (mendukung evaluasi langkah matematis SymPy)
python evaluate_model.py --subject economy --ref-file reference_answer_economy.xlsx

# Bahasa Inggris
python evaluate_model.py --subject english --ref-file reference_answer_english.xlsx

# Sejarah
python evaluate_model.py --subject sejarah --ref-file reference_answer_history.xlsx
```

---

## 5. Panduan Membaca Hasil & Metrik Evaluasi

Setelah skrip selesai berjalan, terminal akan menampilkan metrik statistik performa berikut:

### 1. Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |\text{Skor Guru}_i - \text{Skor AI}_i|$$
* **Makna:** Rata-rata selisih mutlak nilai model AI terhadap nilai guru per butir soal.
* **Interpretasi:** Semakin kecil mendekati 0, semakin akurat penilaian sistem.

### 2. Root Mean Squared Error (RMSE)
$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (\text{Skor Guru}_i - \text{Skor AI}_i)^2}$$
* **Makna:** Sensitif terhadap kesalahan besar (*outlier*). Jika RMSE mendekati MAE, menandakan sistem tidak menghasilkan error ekstrem.

### 3. Akurasi Sempurna (Exact Match, Error = 0)
* **Makna:** Persentase butir jawaban di mana nilai AI **sama persis (100% identik)** dengan nilai guru ($|\text{Guru} - \text{AI}| = 0$).

### 4. Toleransi Selisih $\le 0.5$ Poin
* **Makna:** Persentase butir jawaban dengan simpangan sangat tipis (maksimal setengah poin). Menggambarkan variasi wajar antar-penilai manusia.

### 5. Toleransi Selisih $\le 1.0$ Poin (*Adjacent Agreement*)
* **Makna:** Persentase jawaban yang berada dalam rentang deviasi maksimal 1.0 poin. Dalam literatur AES (*Automated Essay Scoring*), nilai di atas 90% dikategorikan sangat andal.

### 6. Korelasi Pearson ($r$) & Spearman ($\rho$)
* Mengukur derajat kelinearan ($r$) dan urutan peringkat ($\rho$) antara penilaian AI dan guru pada rentang $-1.0$ s.d. $+1.0$.
* Nilai $r \ge 0.80$ dikategorikan sebagai **Korelasi Sangat Kuat (*Very Strong Correlation*)**.

---

## 6. Lokasi Penyimpanan Output Hasil Uji

Secara otomatis seluruh hasil uji coba disimpan ke folder:
`test_set/test_result/`

1. **`evaluation_<subject>_detailed.xlsx`**:  
   Tabel lengkap per baris jawaban berisi:
   - Nama siswa & ID soal
   - Teks jawaban siswa
   - Nilai guru vs Nilai AI
   - Selisih absolut (*abs_error*) dan kuadrat (*squared_error*)
   - Status autograding (`on` / `off`) dan sumber penilaian (`ai_model` / `teacher_override`).

2. **`evaluation_<subject>_summary.txt`**:  
   Ringkasan teks metrik statistik (MAE, RMSE, Pearson $r$, Spearman $\rho$, persentase toleransi).

> **Catatan Git:** Folder `test_set/test_result/` sudah terdaftar di `.gitignore`, sehingga file-file hasil pengujian tidak akan mengotori riwayat commit repository Anda.

