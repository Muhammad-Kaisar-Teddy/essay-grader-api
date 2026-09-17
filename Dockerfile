# Gunakan image Python ringan
FROM python:3.12-slim

# Mengatur direktori kerja di dalam kontainer
WORKDIR /app

# Menyalin file daftar pustaka (requirements)
COPY requirements.txt .

# Menginstal pustaka Python (tanpa menyimpan cache agar ukuran file kecil)
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin seluruh kode kita ke dalam kontainer
COPY . .

# MENGUNDUH DATA NLP (NLTK) SAAT PROSES BUILD
# Agar saat server menyala pertama kali, ia tidak perlu membuang waktu untuk mengunduh corpus bahasa
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Membuka port 8000 (Standar yang sering digunakan)
EXPOSE 8000

# Perintah yang dijalankan saat server dinyalakan
# Menggunakan Gunicorn (Server Produksi) dipasangkan dengan pekerja Uvicorn (FastAPI)
CMD ["gunicorn", "main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
