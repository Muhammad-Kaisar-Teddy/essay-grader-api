import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------------------------
# Kata-kata negasi: TIDAK boleh dihapus dari stopword removal
# Dipertahankan agar N-gram bisa membentuk token seperti "tidak naik" / "not increase"
# ---------------------------------------------------------------------------
NEGATION_WORDS = {
    # Bahasa Indonesia
    "tidak", "bukan", "tanpa", "jangan", "belum", "tak", "tiada",
    # Bahasa Inggris
    "not", "no", "never", "without", "none", "neither", "nor",
    "cannot", "cant", "wont", "dont", "isnt", "arent", "wasnt", "werent",
    "havent", "hasnt", "hadnt", "shouldnt", "wouldnt", "couldnt",
}

# ---------------------------------------------------------------------------
# Download NLTK resources jika belum ada
# ---------------------------------------------------------------------------
for _resource, _pkg in [
    ('corpora/stopwords', 'stopwords'),
    ('corpora/wordnet', 'wordnet'),
    ('corpora/omw-1.4', 'omw-1.4'),
]:
    try:
        nltk.data.find(_resource)
    except LookupError:
        nltk.download(_pkg, quiet=True)

# ---------------------------------------------------------------------------
# Inisialisasi tools NLP
# ---------------------------------------------------------------------------
_factory = StemmerFactory()
_sastrawi = _factory.create_stemmer()
_lemmatizer = WordNetLemmatizer()

# Stopwords gabungan Bahasa Indonesia + Bahasa Inggris,
# dengan pengecualian untuk kata negasi
_id_stops = set(stopwords.words('indonesian')) - NEGATION_WORDS
_en_stops = set(stopwords.words('english')) - NEGATION_WORDS
STOP_WORDS = _id_stops | _en_stops


def _normalize_word(word: str) -> str:
    """
    Menormalisasi satu kata menggunakan Sastrawi (ID) dan NLTK lemmatizer (EN).
    Pilih hasil yang lebih pendek sebagai bentuk dasar yang lebih tereduksi.
    Kata negasi dikembalikan apa adanya.
    """
    if word in NEGATION_WORDS:
        return word
    stemmed = _sastrawi.stem(word)
    lemmatized = _lemmatizer.lemmatize(word)
    return stemmed if len(stemmed) <= len(lemmatized) else lemmatized


def preprocess_text(text: str) -> str:
    """
    Membersihkan dan memproses teks bilingual (Bahasa Indonesia + Bahasa Inggris).

    Tahapan:
    1. Case folding (lowercase)
    2. Hapus tanda baca & angka
    3. Tokenisasi
    4. Stopword removal (negasi dipertahankan)
    5. Stemming (Sastrawi) + Lemmatization (NLTK WordNet) — pilih yang lebih pendek
    """
    if not text:
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Hapus tanda baca & angka
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 3. Tokenisasi & 4. Stopword removal (pertahankan negasi)
    tokens = [
        word for word in text.split()
        if word in NEGATION_WORDS or word not in STOP_WORDS
    ]

    # 5. Normalisasi (stem/lemmatize)
    processed = [_normalize_word(token) for token in tokens]

    return " ".join(processed)
