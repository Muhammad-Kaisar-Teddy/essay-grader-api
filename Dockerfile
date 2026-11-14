# Gunakan image Python 3.12 slim
FROM python:3.12-slim

# Variabel lingkungan untuk Python, path NLTK global, dan cache Hugging Face global
ENV PYTHONUNBUFFERED=1 \
    NLTK_DATA=/usr/share/nltk_data \
    HF_HOME=/usr/share/huggingface

# Mengatur direktori kerja di dalam kontainer
WORKDIR /app

# Menyalin dan menginstal dependensi
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Mengunduh korpus NLTK ke direktori publik (/usr/share/nltk_data) agar bisa diakses semua user
RUN python -c "import nltk; nltk.download('stopwords', download_dir='/usr/share/nltk_data'); nltk.download('wordnet', download_dir='/usr/share/nltk_data'); nltk.download('omw-1.4', download_dir='/usr/share/nltk_data')"

# Mengunduh model nlp-id saat proses build agar tidak membuang waktu dan memori saat runtime
RUN python -c "from nlp_id import PosTag; PosTag()"

# Membuat user standar UID 1000 (Standar keamanan Render & Hugging Face)
RUN useradd -m -u 1000 user && \
    chown -R user:user /app && \
    chown -R user:user /usr/share/huggingface

# Menyalin seluruh kode dengan hak kepemilikan user 1000
COPY --chown=user:user . .

USER user

# Membuka port 10000 (Port standar Render Web Service)
EXPOSE 10000

# Menjalankan Uvicorn langsung (1 worker super ringan ~200MB RAM, bebas OOM dan responsif di Render Free Tier)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
