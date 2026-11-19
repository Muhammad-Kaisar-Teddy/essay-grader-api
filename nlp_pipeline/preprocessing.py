import html
import re
import string
import warnings
from functools import lru_cache
from typing import List, Tuple, Optional, Any


# Suppress unpickling / sklearn version warnings from legacy model bundles
warnings.filterwarnings("ignore", message=".*HF Hub.*")
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk.corpus import stopwords
from num2words import num2words

# Import bilingual NLP models: nlp-id (Indonesian) & spaCy (English)
from nlp_id.lemmatizer import Lemmatizer as IdLemmatizer
from nlp_id.postag import PosTag as IdPosTag
import spacy

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
# Download NLTK resources jika belum ada (aman untuk mode offline)
# ---------------------------------------------------------------------------
for _resource, _pkg in [
    ('corpora/stopwords', 'stopwords'),
    ('corpora/wordnet', 'wordnet'),
    ('corpora/omw-1.4', 'omw-1.4'),
    ('tokenizers/punkt', 'punkt'),
    ('tokenizers/punkt_tab', 'punkt_tab'),
]:
    try:
        nltk.data.find(_resource)
    except LookupError:
        try:
            nltk.data.find(f"{_resource}.zip")
        except LookupError:
            pass

# ---------------------------------------------------------------------------
# Inisialisasi tools NLP Bilingual
# ---------------------------------------------------------------------------
_factory = StemmerFactory()
_sastrawi = _factory.create_stemmer()

# Lemmatizer dan POS Tagger Bahasa Indonesia (nlp-id, berbasis kamus resmi KBBI)
_id_lemmatizer = IdLemmatizer()
_id_postagger: Optional[IdPosTag] = None


def _get_id_postagger() -> IdPosTag:
    global _id_postagger
    if _id_postagger is None:
        _id_postagger = IdPosTag()
    return _id_postagger

# Model Bahasa Inggris spaCy (lemmatizer + POS tagger)
_en_nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# ---------------------------------------------------------------------------
# Kata-kata bilangan: TIDAK boleh dihapus dari stopword removal
# Dipertahankan agar data numerik esai (misal: "5 sila", "3 pilar") tidak lenyap
# ---------------------------------------------------------------------------
NUMBER_WORDS = {
    # Bahasa Indonesia
    'nol', 'satu', 'dua', 'tiga', 'empat', 'lima', 'enam', 'tujuh', 'delapan', 'sembilan', 'sepuluh',
    'sebelas', 'belas', 'puluh', 'ratus', 'ribu', 'juta',
    # Bahasa Inggris
    'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'eleven', 'twelve', 'hundred', 'thousand', 'million'
}

# Stopwords gabungan Bahasa Indonesia + Bahasa Inggris,
# dengan pengecualian untuk kata negasi dan kata bilangan
_id_stops = (set(stopwords.words('indonesian')) - NEGATION_WORDS) - NUMBER_WORDS
_en_stops = (set(stopwords.words('english')) - NEGATION_WORDS) - NUMBER_WORDS
STOP_WORDS = _id_stops | _en_stops


@lru_cache(maxsize=10000)
def _lemmatize_id_word(word: str) -> str:
    """Lemmatisasi kata tunggal bahasa Indonesia menggunakan nlp-id."""
    try:
        return _id_lemmatizer.lemmatize(word)
    except Exception:
        return word


@lru_cache(maxsize=10000)
def _lemmatize_en_word(word: str) -> str:
    """Lemmatisasi kata tunggal bahasa Inggris menggunakan spaCy."""
    try:
        doc = _en_nlp(word)
        return doc[0].lemma_.lower() if doc else word
    except Exception:
        return word


@lru_cache(maxsize=10000)
def _stem_id_word(word: str) -> str:
    """Stemming fallback bahasa Indonesia menggunakan Sastrawi dengan cache."""
    try:
        return _sastrawi.stem(word)
    except Exception:
        return word


def _is_known_english_word(word: str) -> bool:
    """Periksa apakah kata terdaftar dalam kosakata bahasa Inggris (spaCy string store)."""
    if not word:
        return False
    return word in _en_nlp.vocab.strings or word.lower() in _en_nlp.vocab.strings


_NUMERIC_STR_RE = re.compile(r'^\d+(?:\.\d+)?$')


@lru_cache(maxsize=20000)
def _normalize_word(word: str, lang: Optional[str] = None) -> str:
    """
    Menormalisasi satu kata menggunakan Lemmatizer Bilingual yang sadar bahasa (language-aware):
    1. Negasi, kata bilangan, dan digit/desimal dikembalikan apa adanya.
    2. Jika `lang == 'en'`: prioritaskan spaCy EN, dan hanya fallback ke `nlp-id` jika ada kata
       sisipan ID (tanpa memanggil Sastrawi pada kata bahasa Inggris).
    3. Jika `lang == 'id'` atau `lang is None`: prioritaskan `nlp-id`, lindungi kata dasar baku KBBI (`root_word`),
       lalu gunakan spaCy untuk kata bahasa Inggris/code-mixing (tanpa memanggil Sastrawi pada kata Inggris).
    """
    if word in NEGATION_WORDS or word in NUMBER_WORDS or word.isdigit() or _NUMERIC_STR_RE.match(word):
        return word

    if lang == "en":
        # 1. Lemmatizer Bahasa Inggris (spaCy)
        en_lemma = _lemmatize_en_word(word)
        if en_lemma != word:
            return en_lemma

        # 2. Jika ada kata sisipan Bahasa Indonesia (code-mixing)
        id_lemma = _lemmatize_id_word(word)
        if id_lemma != word:
            return id_lemma

        return word

    # Untuk lang == "id" atau lang is None:
    # 1. Lemmatizer Bahasa Indonesia (nlp-id)
    id_lemma = _lemmatize_id_word(word)
    if id_lemma != word:
        return id_lemma

    # 2. Pelindung kata dasar KBBI: jika sudah merupakan kata dasar baku (mis. "boros", "kardus"),
    #    jangan biarkan lemmatizer bahasa Inggris (spaCy) atau stemmer lain memotong akhiran '-s'.
    if word in _id_lemmatizer.root_word:
        return word

    # 3. Lemmatizer Bahasa Inggris (spaCy) untuk kata asing/code-mixing
    known_en = _is_known_english_word(word)
    if lang is None or known_en:
        en_lemma = _lemmatize_en_word(word)
        if en_lemma != word:
            return en_lemma
        if known_en:
            return word

    # 4. Fallback: jika Sastrawi menghasilkan bentuk yang lebih pendek
    stemmed = _stem_id_word(word)
    if len(stemmed) < len(word):
        return stemmed

    return word


# Hanya tag HTML yang DIKENAL yang dihapus; atribut harus berbentuk key=value.
# Jadi "<b dan c>" (teks siswa) TIDAK dianggap tag, sedangkan <a href="x"> tetap.
_KNOWN_TAGS = (
    r"(?:a|abbr|b|blockquote|br|code|div|em|font|h[1-6]|hr|i|img|li|mark|"
    r"ol|p|pre|s|small|span|strike|strong|sub|sup|table|tbody|td|th|"
    r"thead|tr|u|ul)"
)
_ATTRS = r"(?:\s+[A-Za-z_:][\w:.\-]*\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s\"'<>=]+))*"
_TAG_RE = re.compile(r"</?" + _KNOWN_TAGS + r"\b" + _ATTRS + r"\s*/?>", re.I)
_BR_RE = re.compile(r"<br\b" + _ATTRS + r"\s*/?>", re.I)
_BLOCK_CLOSE_RE = re.compile(r"</(?:p|div|li|tr|h[1-6]|blockquote)\s*>", re.I)


def clean_html_and_newlines(text: str) -> str:
    """
    Membersihkan tag HTML dan menormalisasi jeda baris (Enter / Shift + Enter).

    1. Mengubah tag pemisah baris/blok HTML (<br>, <p>, <div>, <li>, dll.) menjadi newline '\\n'.
       - Shift + Enter di web editor menghasilkan <br> atau <br/>.
       - Enter di web editor menghasilkan blok baru seperti <p>...</p> atau <div>...</div>.
    2. Menangani karakter newline mentah (\\r\\n, \\r, unicode line/paragraph separator \\u2028, \\u2029).
    3. Menghapus komentar HTML dan tag HTML yang dikenal saja (misal: <b>, <i>, <span>),
       sehingga simbol '<' dan '>' pada ekspresi matematika/informatika (misal: a<b dan c>d) tetap aman.
    4. Decode HTML entities (misal: &nbsp; menjadi spasi, &lt; menjadi <, &gt; menjadi >, dsb.).
    5. Merapikan spasi dan baris kosong berlebih.
    """
    if not text:
        return ""

    # 1. Normalisasi newline sistem dan unicode line/paragraph separator
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = text.replace('\u2028', '\n').replace('\u2029', '\n')

    # 2. Shift + Enter: <br>, <br/>, <br /> -> \n
    text = _BR_RE.sub('\n', text)

    # 3. Enter (blok penutup paragraf/div/list/heading/table row) -> \n
    text = _BLOCK_CLOSE_RE.sub('\n', text)

    # 4. Hapus komentar HTML dan tag HTML yang dikenal (buka/inline/tutup)
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    text = _TAG_RE.sub('', text)

    # 5. Decode HTML entities (&nbsp; -> space, &lt; -> <, &gt; -> >, dll)
    text = html.unescape(text)
    text = text.replace('\xa0', ' ')

    # 6. Bersihkan spasi horizontal berlebih di tiap baris dan rapikan \n
    lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.split('\n')]

    cleaned_lines = []
    prev_empty = False
    for line in lines:
        if line == '':
            if not prev_empty:
                cleaned_lines.append('')
                prev_empty = True
        else:
            cleaned_lines.append(line)
            prev_empty = False

    return '\n'.join(cleaned_lines).strip()


# ---------------------------------------------------------------------------
# Kamus Kontraksi Bahasa Inggris (Contraction Expansion)
# ---------------------------------------------------------------------------
CONTRACTIONS = {
    # Bentuk Negasi
    "can't": "can not",
    "cannot": "cannot",
    "cant": "can not",
    "won't": "will not",
    "wont": "will not",
    "shan't": "shall not",
    "shant": "shall not",
    "don't": "do not",
    "dont": "do not",
    "doesn't": "does not",
    "doesnt": "does not",
    "didn't": "did not",
    "didnt": "did not",
    "isn't": "is not",
    "isnt": "is not",
    "aren't": "are not",
    "arent": "are not",
    "wasn't": "was not",
    "wasnt": "was not",
    "weren't": "were not",
    "werent": "were not",
    "haven't": "have not",
    "havent": "have not",
    "hasn't": "has not",
    "hasnt": "has not",
    "hadn't": "had not",
    "hadnt": "had not",
    "wouldn't": "would not",
    "wouldnt": "would not",
    "shouldn't": "should not",
    "shouldnt": "should not",
    "couldn't": "could not",
    "couldnt": "could not",
    "mustn't": "must not",
    "mustnt": "must not",
    "ain't": "am not",
    "aint": "am not",

    # Pronoun / Noun + be / have / will / would
    "i'm": "i am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "they're": "they are",
    "that's": "that is",
    "there's": "there is",
    "here's": "here is",
    "what's": "what is",
    "who's": "who is",
    "where's": "where is",
    "when's": "when is",
    "why's": "why is",
    "how's": "how is",

    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "could've": "could have",
    "should've": "should have",
    "would've": "would have",
    "might've": "might have",
    "must've": "must have",

    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "we'll": "we will",
    "they'll": "they will",
    "it'll": "it will",
    "that'll": "that will",

    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would",

    "let's": "let us",
}

_SORTED_CONTRACTIONS = sorted(CONTRACTIONS.keys(), key=len, reverse=True)
_CONTRACTIONS_RE = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in _SORTED_CONTRACTIONS) + r')\b',
    flags=re.IGNORECASE
)


def expand_contractions(text: str) -> str:
    """
    Mengembangkan singkatan (contractions) bahasa Inggris ke bentuk panjangnya.
    Contoh: "don't" -> "do not", "it's" -> "it is", "they're" -> "they are".
    Juga menormalisasi berbagai varian kutip tunggal/curly apostrophes (’ ‘ ´ `).
    """
    if not text:
        return ""
    # Normalisasi berbagai jenis karakter apostrof
    text = re.sub(r"[’‘´`]", "'", text)
    return _CONTRACTIONS_RE.sub(
        lambda m: CONTRACTIONS.get(m.group(0).lower(), m.group(0)),
        text
    )


_expand_contractions = expand_contractions



# ---------------------------------------------------------------------------
# Deteksi Bahasa Sederhana & Konversi Angka ke Kata Selektif (Num2Words)
# ---------------------------------------------------------------------------
_ID_INDICATORS = {
    'dan', 'yang', 'di', 'dari', 'ini', 'itu', 'adalah', 'untuk', 'dengan',
    'pada', 'oleh', 'ke', 'karena', 'bisa', 'dapat', 'akan', 'atau', 'telah'
}
_EN_INDICATORS = {
    'the', 'and', 'is', 'are', 'in', 'of', 'to', 'for', 'with', 'on', 'by',
    'this', 'that', 'from', 'as', 'at', 'an', 'be', 'has', 'have', 'can'
}


def detect_language(text: str) -> str:
    """
    Deteksi bahasa dominan teks: 'id' (Bahasa Indonesia) atau 'en' (Bahasa Inggris).
    1. Mencocokkan fungsi kata (stopword/function words) utama kedua bahasa.
    2. Jika seimbang / tanpa kata fungsi (misal esai singkat teknis), gunakan perbandingan
       kosakata bahasa Inggris (WordNet) vs bahasa Indonesia (kamus morfologi nlp-id).
    """
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    if not words:
        return 'id'

    unique_words = set(words)
    en_matches = len(unique_words & _EN_INDICATORS)
    id_matches = len(unique_words & _ID_INDICATORS)

    if en_matches > id_matches:
        return 'en'
    elif id_matches > en_matches:
        return 'id'

    # Fallback jika skor indikator seimbang: cek leksikon ID (nlp-id) vs EN (WordNet / spaCy vocab)
    id_count = sum(1 for w in unique_words if w in _id_lemmatizer.root_word or w in _id_lemmatizer.lemma_dict)
    try:
        from nltk.corpus import wordnet
        en_count = sum(1 for w in unique_words if wordnet.synsets(w))
    except LookupError:
        en_count = sum(1 for w in unique_words if _is_known_english_word(w))

    return 'en' if en_count > id_count else 'id'



# Angka utuh = tidak menempel huruf/angka lain, dan titik/koma di belakangnya
# hanya boleh jika BUKAN pemisah desimal/ribuan (tidak diikuti digit).
_INT_RE = re.compile(r'(?<![\w.,])(\d+)(?!\w|[.,]\d)')


def convert_numbers_to_words(text: str, max_val: int = 100, lang: Optional[str] = None) -> str:
    """
    Mengonversi bilangan bulat (0 s.d max_val, default 100) menjadi kata secara selektif.
    - Mendukung bilingual (Bahasa Indonesia & Bahasa Inggris secara otomatis).
    - Selektif: mengabaikan baris yang mengandung persamaan matematika ('='),
      mengabaikan angka desimal (misal: 0.5, 3.14), variabel aljabar (misal: 2x, P1),
      dan angka tahun/angka besar di atas max_val (misal: 1945, 2024).
    - Tetap mengonversi angka di akhir kalimat/klausa yang diikuti tanda baca (misal: "ada 5.").
    """
    if not text:
        return ""

    if lang is None:
        lang = detect_language(text)

    lines = text.split('\n')
    converted_lines = []

    for line in lines:
        # Jika baris adalah persamaan matematika, pertahankan angka untuk SymPy / perhitungan eksak
        if '=' in line:
            converted_lines.append(line)
            continue

        def _num_repl(match):
            val_str = match.group(1)
            try:
                val = int(val_str)
                if 0 <= val <= max_val:
                    return num2words(val, lang=lang)
            except Exception:
                pass
            return val_str

        converted_line = _INT_RE.sub(_num_repl, line)
        converted_lines.append(converted_line)

    return '\n'.join(converted_lines)


def strip_punctuation(text: str) -> str:
    """
    Menghapus/mengganti tanda baca secara sadar-konteks:
    - Apostrof (' dan `) dihapus tanpa spasi (rina's -> rinas).
    - Titik (.) dan koma (,) di antara digit dipertahankan (0,5 / 3.14 / 30.000).
    - Tanda baca lain (-, /, (, ), ,, ., dll.) diganti spasi agar kata tidak menyatu
      (misal: 'kata-kata' -> 'kata kata', 'dan/atau' -> 'dan atau').
    """
    out = []
    n = len(text)
    for i, ch in enumerate(text):
        if ch not in string.punctuation:
            out.append(ch)
        elif ch in ".," and 0 < i < n - 1 and text[i - 1].isdigit() and text[i + 1].isdigit():
            out.append(ch)
        elif ch in "'`":
            continue
        else:
            out.append(" ")
    return "".join(out)


_NUM_TOKEN_RE = re.compile(r'\b\d+(?:[.,]\d+)*\b')
_ID_THOUSANDS = re.compile(r'^\d{1,3}(?:\.\d{3})+$')   # 30.000 / 1.250.000
_EN_THOUSANDS = re.compile(r'^\d{1,3}(?:,\d{3})+$')    # 30,000


def normalize_numeric_token(tok: str) -> str:
    """Menormalisasi format angka: 30.000 / 30,000 -> 30000; 0,5 -> 0.5; 3.14 -> 3.14."""
    if _ID_THOUSANDS.match(tok):
        return tok.replace('.', '')
    if _EN_THOUSANDS.match(tok):
        return tok.replace(',', '')
    return tok.replace(',', '.')


def filter_numbers(text: str, preserve_years: bool = True, max_val: int = 100) -> str:
    """
    Menyaring angka sisa setelah penghapusan tanda baca:
    - Bilangan bulat kecil (<= max_val) yang belum jadi kata (misal convert_numbers=False) dibuang.
    - Tahun 4-digit (1000-2099) dipertahankan hanya bila preserve_years=True.
    - Angka desimal (0.5, 3.14), ribuan (30000), dan angka besar (> max_val, misal 250) dipertahankan.
    """
    def _repl(match):
        tok = normalize_numeric_token(match.group(0))
        if tok.isdigit():
            v = int(tok)
            if len(tok) == 4 and 1000 <= v <= 2099:
                return tok if preserve_years else ""
            if v <= max_val:
                return ""
        return tok

    return _NUM_TOKEN_RE.sub(_repl, text)


def preprocess_text(
    text: str,
    clean_html: bool = True,
    case_folding: bool = True,
    expand_contractions: bool = True,
    convert_numbers: bool = True,
    remove_punctuation: bool = True,
    preserve_years: bool = True,
    remove_stopwords: bool = True,
    use_lemmatization: bool = True,
    flags: Optional[Any] = None
) -> str:
    """
    Membersihkan dan memproses teks bilingual dengan Feature Flags fleksibel.

    Mendukung konfigurasi On/Off untuk masing-masing tahap preprocessing (Ablation Study):
    - clean_html: Pembersihan tag HTML & normalisasi jeda baris
    - case_folding: Normalisasi huruf kecil (lowercase)
    - expand_contractions: Ekspansi singkatan bahasa Inggris (don't -> do not)
    - convert_numbers: Konversi bilangan 0-100 ke kata (num2words)
    - remove_punctuation: Penghapusan tanda baca
    - preserve_years: Preservasi angka tahun 4-digit rentang 1000 - 2099
    - remove_stopwords: Pembersihan kata henti (negasi & bilangan dilindungi)
    - use_lemmatization: Lemmatisasi bilingual (nlp-id untuk ID, spaCy untuk EN)
    - flags: Objek PreprocessingFlags atau dict untuk mengeset seluruh flag sekaligus.
    """
    if flags is not None:
        if hasattr(flags, "model_dump"):
            d = flags.model_dump()
        elif isinstance(flags, dict):
            d = flags
        else:
            d = getattr(flags, "__dict__", {})
        clean_html = d.get("clean_html", clean_html)
        case_folding = d.get("case_folding", case_folding)
        expand_contractions = d.get("expand_contractions", expand_contractions)
        convert_numbers = d.get("convert_numbers", convert_numbers)
        remove_punctuation = d.get("remove_punctuation", remove_punctuation)
        preserve_years = d.get("preserve_years", preserve_years)
        remove_stopwords = d.get("remove_stopwords", remove_stopwords)
        use_lemmatization = d.get("use_lemmatization", use_lemmatization)

    if not text:
        return ""

    return _preprocess_text_cached(
        text,
        bool(clean_html),
        bool(case_folding),
        bool(expand_contractions),
        bool(convert_numbers),
        bool(remove_punctuation),
        bool(preserve_years),
        bool(remove_stopwords),
        bool(use_lemmatization),
    )


@lru_cache(maxsize=8192)
def _preprocess_text_cached(
    text: str,
    clean_html: bool,
    case_folding: bool,
    expand_contractions: bool,
    convert_numbers: bool,
    remove_punctuation: bool,
    preserve_years: bool,
    remove_stopwords: bool,
    use_lemmatization: bool,
) -> str:
    # 0. Normalisasi HTML & jeda baris
    if clean_html:
        text = clean_html_and_newlines(text)

    # Deteksi bahasa dominan teks sebelum normalisasi / lemmatisasi
    detected_lang = detect_language(text)

    # 1. Lowercase (Case folding)
    if case_folding:
        text = text.lower()

    # 2. Ekspansi Kontraksi Bahasa Inggris
    if expand_contractions:
        text = _expand_contractions(text)


    # 3. Konversi Angka ke Kata Selektif
    if convert_numbers:
        text = convert_numbers_to_words(text, lang=detected_lang)

    # 4. Hapus tanda baca & saring angka sisa
    if remove_punctuation:
        text = strip_punctuation(text)
    
    # filter_numbers harus selalu dipanggil jika preserve_years=False, 
    # meskipun remove_punctuation=False (misal pada mode 'exact').
    if not preserve_years or remove_punctuation:
        text = filter_numbers(text, preserve_years=preserve_years)

    # 5. Tokenisasi & Stopword removal
    tokens = text.split()
    if remove_stopwords:
        tokens = [
            word for word in tokens
            if (
                word.isdigit()
                or _NUMERIC_STR_RE.match(word)
                or word in NEGATION_WORDS
                or word in NUMBER_WORDS
                or word not in STOP_WORDS
            )
        ]

    # 6. Normalisasi (Lemmatisasi bilingual sadar-bahasa)
    if use_lemmatization:
        tokens = [_normalize_word(token, lang=detected_lang) for token in tokens]

    return " ".join(tokens)



def lemmatize_text(text: str) -> str:
    """
    Melakukan lemmatisasi bilingual terhadap teks secara utuh.
    Mengonversi kata inflektif (ID & EN) ke bentuk dasarnya (lemma)
    tanpa menghapus stopwords atau merusak tanda baca.
    """
    if not text:
        return ""

    text = clean_html_and_newlines(text)
    detected_lang = detect_language(text)

    def _repl(match):
        w = match.group(0)
        return _normalize_word(w.lower(), lang=detected_lang)

    return re.sub(r'\b[a-zA-Z0-9_-]+\b', _repl, text)


@lru_cache(maxsize=4096)
def get_pos_tags(text: str, refine_foreign: bool = False) -> List[Tuple[str, str]]:
    """
    Melakukan Part-of-Speech (POS) Tagging pada teks bilingual (Bahasa Indonesia & Bahasa Inggris).

    Parameter:
        text: Kalimat / teks esai yang akan diberi tag POS.
        refine_foreign: Jika True, kata yang terdeteksi sebagai kata asing ('FW')
                        oleh tagger Indonesia akan diperjelas dengan kelas kata
                        Bahasa Inggris dari spaCy (misal: 'FW (VERB)', 'FW (NOUN)').

    Mengembalikan:
        List of (token, pos_tag).
        Contoh ID: [('Tumbuhan', 'NN'), ('membuat', 'VB'), ('makanan', 'NN')]
        Contoh EN: [('Plants', 'NOUN'), ('produce', 'VERB'), ('food', 'NOUN')]
    """
    if not text:
        return []

    clean_text = clean_html_and_newlines(text)
    if not clean_text:
        return []

    lang = detect_language(clean_text)

    if lang == 'en':
        doc = _en_nlp(clean_text)
        return [(token.text, token.pos_) for token in doc if not token.is_space]
    else:
        raw_tags = _get_id_postagger().get_pos_tag(clean_text)
        if not refine_foreign:
            return raw_tags

        refined_tags = []
        for token, tag in raw_tags:
            if tag == 'FW':
                doc = _en_nlp(token)
                en_pos = doc[0].pos_ if doc else 'X'
                refined_tags.append((token, f"FW ({en_pos})"))
            else:
                refined_tags.append((token, tag))
        return refined_tags

