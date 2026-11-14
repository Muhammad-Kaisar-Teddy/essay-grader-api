# Gunakan image Python 3.12 slim
FROM python:3.12-slim

# Variabel lingkungan untuk Python dan path NLTK global
ENV PYTHONUNBUFFERED=1 \
    NLTK_DATA=/usr/share/nltk_data

# Mengatur direktori kerja di dalam kontainer
WORKDIR /app

# Menyalin dan menginstal dependensi
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Mengunduh korpus NLTK ke direktori publik (/usr/share/nltk_data) agar bisa diakses semua user
RUN python -c "import nltk; nltk.download('stopwords', download_dir='/usr/share/nltk_data'); nltk.download('wordnet', download_dir='/usr/share/nltk_data'); nltk.download('omw-1.4', download_dir='/usr/share/nltk_data')"

# Membuat user standar UID 1000 (Standar keamanan Hugging Face Spaces)
RUN useradd -m -u 1000 user && \
    chown -R user:user /app

# Menyalin seluruh kode dengan hak kepemilikan user 1000
COPY --chown=user:user . .

USER user

# Membuka port 7860 dan port default Render
EXPOSE 7860

# Menjalankan Gunicorn dengan 2 worker (hemat RAM agar fit di 512MB Render)
# Menggunakan ${PORT:-7860} agar otomatis menyesuaikan port Render ($PORT) maupun default (7860)
CMD ["sh", "-c", "gunicorn main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-7860}"]
