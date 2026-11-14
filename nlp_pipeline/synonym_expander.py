"""
Modul Ekspansi Sinonim & Penyelarasan Token Kanonik (WordNet + Tesaurus Bilingual).

Fungsi utama:
1. POS-Guided Synset Lookup:
   Mencari sinonim kata/frasa (Bahasa Indonesia & Bahasa Inggris) berdasarkan
   kelompok synset terkurasi dan NLTK WordNet (jika tersedia di lokal), dengan
   filter kelas kata (POS tag) untuk mencegah ambiguitas polisemi (misal:
   "bisa" = mampu vs racun, "bunga" = kembang vs suku bunga).
2. Rule-Based Auto Keyword Expansion:
   Memperluas kata kunci non-strict secara otomatis tanpa merusak kata kunci
   strict (**), kata negasi, atau angka.
3. LSA Canonical Token Alignment:
   Menyelaraskan token sinonim pada jawaban siswa ke token kanonik yang ada
   pada kunci jawaban referensi sebelum pembentukan matriks TF-IDF + SVD,
   sehingga mencegah masalah Vocabulary Mismatch pada LSA tanpa mengubah
   perilaku penalti negasi atau angka.
"""

from typing import List, Set, Dict, Optional, Any, Tuple
from nlp_pipeline.preprocessing import (
    preprocess_text,
    get_pos_tags,
    NEGATION_WORDS,
    NUMBER_WORDS,
    STOP_WORDS,
    _normalize_word,
    _id_lemmatizer,
)

# Coba muat WordNet dari NLTK secara aman (tanpa memicu download paksa saat offline)
_WORDNET_AVAILABLE = False
try:
    import nltk
    from nltk.corpus import wordnet as wn
    # Uji apakah korpus wordnet benar-benar terpasang di disk lokal
    nltk.data.find("corpora/wordnet")
    _WORDNET_AVAILABLE = True
except Exception:
    _WORDNET_AVAILABLE = False

_OMW_AVAILABLE = False
if _WORDNET_AVAILABLE:
    try:
        nltk.data.find("corpora/omw-1.4")
        _OMW_AVAILABLE = True
    except Exception:
        _OMW_AVAILABLE = False


# ==============================================================================
# DAFTAR KATA AMBIGU / POLISEMI (Dibatasi agar tidak salah ekspansi lintas makna)
# ==============================================================================
# Kata-kata ini memiliki lebih dari satu makna yang bertolak belakang atau berbeda
# domain (misal "bisa" = dapat vs racun; "bunga" = tanaman vs bank; "kali" = sungai vs perkalian;
# "kabur" = melarikan diri vs buram; "pasar" = tempat jual beli vs segmen).
POLYSEMOUS_GUARD_WORDS: Set[str] = {
    "bisa", "bunga", "kali", "kabur", "bulan", "mata", "buah", "daun",
    "akar", "kepala", "kaki", "tangan", "hati", "jatuh", "bangun",
    "buku", "halaman", "rapat", "tahu", "apel", "mental", "sedang",
    "hak", "uang", "modal",
}


# ==============================================================================
# TESAURUS SINONIM BILINGUAL TERKURASI (INDONESIA & INGGRIS)
# ==============================================================================
# Setiap entri adalah tuple (pos_category, set_of_synonyms).
# pos_category: "VB" (Verba), "JJ" (Adjektiva), "NN" (Nomina), atau "ANY".
# Kata-kata di dalam setiap synset akan dinormalisasi ke bentuk dasar (stem/lemma)
# saat inisialisasi indeks agar cocok dengan output `preprocess_text()`.
RAW_SYNSET_CLUSTERS: List[Tuple[str, Set[str]]] = [
    # --- VERBA (Tindakan / Proses / Perubahan) ---
    ("VB", {"mencemari", "mengotori", "menodai", "mencemarkan", "kontaminasi", "polusi"}),
    ("VB", {"merusak", "menghancurkan", "membinasakan", "meluluhlantakkan", "meruntuhkan", "destruksi"}),
    ("VB", {"meningkat", "bertambah", "naik", "melonjak", "melambung", "menanjak", "meninggi"}),
    ("VB", {"menurun", "berkurang", "turun", "merosot", "menyusut", "anjlok", "jatuh"}),
    ("VB", {"membuat", "menciptakan", "menghasilkan", "memproduksi", "menyusun", "membentuk"}),
    ("VB", {"menggunakan", "memakai", "memanfaatkan", "mengaplikasikan", "menerapkan"}),
    ("VB", {"membantu", "menolong", "mendukung", "menunjang", "memfasilitasi"}),
    ("VB", {"menjaga", "melindungi", "memelihara", "melestarikan", "merawat", "mengamankan"}),
    ("VB", {"mengawasi", "memantau", "mengontrol", "mengendalikan", "memonitor"}),
    ("VB", {"mengatur", "mengelola", "menata", "mengorganisir", "manajemen"}),
    ("VB", {"mencegah", "menangkal", "menghindari", "menghalangi", "menahan", "preventif"}),
    ("VB", {"mengatasi", "menyelesaikan", "menangani", "menanggulangi", "memecahkan"}),
    ("VB", {"menyebabkan", "mengakibatkan", "menimbulkan", "memicu", "berdampak", "berefek"}),
    ("VB", {"menjelaskan", "menerangkan", "memaparkan", "menguraikan", "mendeskripsikan", "menggambarkan"}),
    ("VB", {"membeli", "memperoleh", "mendapatkan", "belanja", "berbelanja"}),
    ("VB", {"menjual", "memasarkan", "mendistribusikan", "meniagakan", "memperdagangkan"}),
    ("VB", {"menghemat", "menabung", "menyimpan", "menyisihkan"}),
    ("VB", {"menghabiskan", "memboroskan", "menghamburkan", "menguras"}),
    ("VB", {"berbicara", "berkata", "mengucapkan", "bertutur", "berujar", "berkomunikasi"}),
    ("VB", {"melihat", "memandang", "mengamati", "menyaksikan", "menatap", "memperhatikan"}),
    ("VB", {"berpikir", "merenung", "menalar", "mempertimbangkan"}),
    ("VB", {"berjalan", "melangkah", "bergerak", "melaju"}),
    ("VB", {"berlari", "kebut", "mengebut", "melaju"}),
    ("VB", {"menolak", "menampik", "menentang", "membantah", "menyanggah"}),
    ("VB", {"menerima", "menyetujui", "mengiyakan", "menyepakati"}),
    ("VB", {"mengubah", "mengganti", "memodifikasi", "merombak", "mentransformasi"}),

    # --- ADJEKTIVA (Sifat / Keadaan / Karakteristik) ---
    ("JJ", {"konsumtif", "boros", "royal", "impulsif", "hedonis"}),
    ("JJ", {"hemat", "irit", "cermat", "efisien", "ekonomis", "bijak"}),
    ("JJ", {"mudah", "gampang", "praktis", "simpel", "sederhana"}),
    ("JJ", {"sulit", "susah", "rumit", "kompleks", "sukar", "pelik"}),
    ("JJ", {"cepat", "kilat", "lekas", "tangkas", "sigap", "instan", "segera"}),
    ("JJ", {"lambat", "pelan", "lamban", "lelet"}),
    ("JJ", {"besar", "raksasa", "luas", "akbar", "raya", "masif", "agung"}),
    ("JJ", {"kecil", "sempit", "mini", "mungil", "mikro", "minim"}),
    ("JJ", {"indah", "cantik", "elok", "permai", "asri", "bagus", "menawan", "estetis"}),
    ("JJ", {"buruk", "jelek", "rusak", "bobrok", "parah"}),
    ("JJ", {"bersih", "suci", "higienis", "steril", "murni", "jernih"}),
    ("JJ", {"kotor", "cemar", "jorok", "kumuh", "keruh", "tercemar"}),
    ("JJ", {"sopan", "santun", "ramah", "beradab", "beretika", "hormat", "halus"}),
    ("JJ", {"kasar", "angkuh", "sombong", "congkak", "arogan"}),
    ("JJ", {"rindu", "kangen"}),
    ("JJ", {"sedih", "pilu", "duka", "murung", "lara", "pedih", "nestapa"}),
    ("JJ", {"senang", "gembira", "bahagia", "riang", "sukacita", "ceria", "girang"}),
    ("JJ", {"takut", "cemas", "khawatir", "gelisah", "gentar", "ngeri", "waswas"}),
    ("JJ", {"marah", "murka", "geram", "kesal", "dongkol", "emosi", "berang"}),
    ("JJ", {"tenang", "damai", "tenteram", "hening", "sunyi", "sepi", "senyap", "teduh"}),
    ("JJ", {"ramai", "bising", "gaduh", "riuh", "ribut", "semarak"}),
    ("JJ", {"kuat", "kokoh", "tangguh", "perkasa", "solid", "teguh"}),
    ("JJ", {"lemah", "rapuh", "ringkih", "lesu", "lemas", "rentan"}),
    ("JJ", {"kaya", "makmur", "sejahtera", "berkecukupan", "mampu"}),
    ("JJ", {"miskin", "melarat", "fakir", "papa", "kekurangan", "sengsara"}),
    ("JJ", {"pintar", "cerdas", "pandai", "cendekia", "mahir", "cakap", "piawai", "terampil"}),
    ("JJ", {"bodoh", "bebal", "dungu", "tolol", "awam"}),
    ("JJ", {"penting", "krusial", "vital", "utama", "pokok", "esensial", "mendasar"}),
    ("JJ", {"aman", "selamat", "terlindungi", "kondusif", "terjamin"}),
    ("JJ", {"bahaya", "berbahaya", "rawan", "mengancam", "berisiko", "kritis"}),
    ("JJ", {"asli", "otentik", "orisinil", "murni", "tulen", "nyata", "faktual"}),
    ("JJ", {"palsu", "tiruan", "imitasi", "bohong", "fiktif", "semu"}),
    ("JJ", {"modern", "canggih", "mutakhir", "maju", "kontemporer", "digital"}),
    ("JJ", {"tradisional", "kuno", "klasik", "konvensional", "lama", "adat"}),

    # --- NOMINA (Konsep / Entitas / Ekonomi / Bahasa / Sains / Sejarah) ---
    ("NN", {"keuntungan", "laba", "profit", "manfaat", "faedah", "guna", "maslahat", "kelebihan", "keunggulan"}),
    ("NN", {"kerugian", "rugi", "kelemahan", "kekurangan", "keburukan", "defisit", "dampak buruk"}),
    ("NN", {"masalah", "problem", "persoalan", "kendala", "hambatan", "rintangan", "tantangan", "gangguan"}),
    ("NN", {"solusi", "jalan keluar", "penyelesaian", "pemecahan", "penanggulangan", "alternatif"}),
    ("NN", {"aturan", "peraturan", "regulasi", "kebijakan", "ketentuan", "norma", "hukum", "undang-undang"}),
    ("NN", {"tujuan", "sasaran", "target", "maksud", "visi", "cita-cita", "arah"}),
    ("NN", {"alasan", "sebab", "penyebab", "faktor", "latar belakang", "motif", "dasar"}),
    ("NN", {"akibat", "dampak", "efek", "konsekuensi", "imbas", "pengaruh", "hasil"}),
    ("NN", {"tumbuhan", "tanaman", "flora", "pepohonan", "vegetasi", "nabati"}),
    ("NN", {"hewan", "binatang", "satwa", "fauna", "margasatwa"}),
    ("NN", {"manusia", "orang", "individu", "insan", "masyarakat", "warga", "penduduk", "rakyat"}),
    ("NN", {"sampah", "limbah", "kotoran", "polutan", "residu", "buangan"}),
    ("NN", {"lingkungan", "alam", "ekosistem", "habitat", "ekologi"}),
    ("NN", {"hutan", "rimba", "belantara", "alas"}),
    ("NN", {"laut", "samudra", "bahari", "perairan", "pesisir", "pantai"}),
    ("NN", {"angin", "bayu", "udara", "hawa", "atmosfer"}),
    ("NN", {"pendapatan", "penghasilan", "pemasukan", "gaji", "upah", "omzet"}),
    ("NN", {"pengeluaran", "biaya", "ongkos", "beban", "belanja", "tarif", "anggaran"}),
    ("NN", {"harga", "nilai", "tarif", "banderol", "kurs"}),
    ("NN", {"barang", "produk", "komoditas", "benda", "logistik", "objek"}),
    ("NN", {"jasa", "layanan", "pelayanan", "servis"}),
    ("NN", {"pedagang", "penjual", "produsen", "wirausaha", "pengusaha", "pelaku usaha", "niagawan"}),
    ("NN", {"pembeli", "konsumen", "pelanggan", "pemakai", "pengguna"}),
    ("NN", {"pekerjaan", "profesi", "mata pencaharian", "karier", "jabatan", "tugas"}),
    ("NN", {"kewajiban", "tanggung jawab", "keharusan", "amanah"}),
    ("NN", {"bantuan", "subsidi", "tunjangan", "donasi", "santunan", "pertolongan"}),
    ("NN", {"kemerdekaan", "kebebasan", "kedaulatan", "otonomi", "emansipasi"}),
    ("NN", {"penjajahan", "kolonialisme", "imperialisme", "penindasan", "pendudukan", "eksploitasi"}),
    ("NN", {"perlawanan", "perjuangan", "pemberontakan", "revolusi", "konflik", "pertempuran", "perang"}),
    ("NN", {"perjanjian", "kesepakatan", "perundingan", "diplomasi", "mufakat", "konsensus", "traktat"}),
    ("NN", {"pemimpin", "tokoh", "ketua", "kepala", "komandan", "penguasa", "raja", "presiden"}),
    ("NN", {"cerita", "kisah", "narasi", "riwayat", "hikayat", "dongeng"}),
    ("NN", {"pesan", "amanat", "nasihat", "petuah", "pelajaran", "moral", "hikmah"}),

    # --- ENGLISH SYNSETS (Verbs, Adjectives, Nouns for Bilingual / English Essays) ---
    ("VB", {"increase", "rise", "grow", "escalate", "surge", "expand", "boost", "raise"}),
    ("VB", {"decrease", "decline", "drop", "fall", "reduce", "diminish", "shrink", "lower"}),
    ("VB", {"create", "make", "produce", "generate", "build", "construct", "develop", "form"}),
    ("VB", {"destroy", "damage", "ruin", "harm", "devastate", "demolish", "wreck", "spoil"}),
    ("VB", {"pollute", "contaminate", "dirty", "taint", "foul"}),
    ("VB", {"protect", "preserve", "conserve", "safeguard", "defend", "guard", "save", "maintain"}),
    ("VB", {"help", "assist", "aid", "support", "facilitate"}),
    ("VB", {"prevent", "stop", "avoid", "hinder", "avert", "deter", "block"}),
    ("VB", {"solve", "resolve", "fix", "address", "tackle", "overcome", "settle"}),
    ("VB", {"cause", "trigger", "lead", "induce", "provoke", "generate", "bring"}),
    ("VB", {"explain", "describe", "clarify", "illustrate", "elaborate", "define", "state"}),
    ("VB", {"use", "utilize", "employ", "apply", "consume"}),
    ("VB", {"buy", "purchase", "acquire", "obtain", "get"}),
    ("VB", {"sell", "market", "trade", "distribute", "vend"}),
    ("VB", {"change", "modify", "alter", "transform", "convert", "adjust", "shift"}),

    ("JJ", {"important", "crucial", "vital", "essential", "significant", "critical", "key", "major", "fundamental"}),
    ("JJ", {"good", "great", "excellent", "beneficial", "positive", "favorable", "advantageous", "fine"}),
    ("JJ", {"bad", "poor", "harmful", "negative", "adverse", "detrimental", "unfavorable", "terrible", "severe"}),
    ("JJ", {"fast", "quick", "rapid", "swift", "speedy", "prompt", "instant"}),
    ("JJ", {"slow", "gradual", "sluggish", "delayed", "unhurried"}),
    ("JJ", {"big", "large", "huge", "massive", "vast", "enormous", "immense", "giant", "great"}),
    ("JJ", {"small", "little", "tiny", "minor", "minimal", "compact", "minute", "slight"}),
    ("JJ", {"easy", "simple", "effortless", "convenient", "practical", "straightforward"}),
    ("JJ", {"hard", "difficult", "complex", "complicated", "tough", "challenging", "demanding"}),
    ("JJ", {"happy", "glad", "joyful", "cheerful", "delighted", "pleased", "content"}),
    ("JJ", {"sad", "unhappy", "sorrowful", "depressed", "miserable", "gloomy", "grief"}),
    ("JJ", {"angry", "mad", "furious", "upset", "annoyed", "irritated", "outraged"}),
    ("JJ", {"afraid", "scared", "frightened", "terrified", "anxious", "worried", "fearful"}),
    ("JJ", {"beautiful", "pretty", "lovely", "gorgeous", "attractive", "scenic", "stunning"}),
    ("JJ", {"ugly", "unattractive", "hideous", "unpleasant"}),
    ("JJ", {"clean", "pure", "hygienic", "spotless", "clear", "sanitary", "fresh"}),
    ("JJ", {"dirty", "filthy", "polluted", "contaminated", "unclean", "messy", "muddy"}),
    ("JJ", {"rich", "wealthy", "affluent", "prosperous", "well-off"}),
    ("JJ", {"poor", "needy", "impoverished", "destitute", "underprivileged"}),
    ("JJ", {"smart", "intelligent", "clever", "bright", "wise", "brilliant", "skilled"}),
    ("JJ", {"strong", "powerful", "sturdy", "robust", "solid", "tough", "resilient"}),
    ("JJ", {"weak", "fragile", "feeble", "frail", "vulnerable", "delicate"}),
    ("JJ", {"safe", "secure", "protected", "harmless"}),
    ("JJ", {"dangerous", "hazardous", "risky", "unsafe", "perilous", "threatening"}),
    ("JJ", {"modern", "advanced", "contemporary", "new", "innovative", "digital"}),
    ("JJ", {"traditional", "ancient", "old", "classic", "conventional", "customary"}),

    ("NN", {"benefit", "advantage", "profit", "gain", "merit", "value", "usefulness"}),
    ("NN", {"loss", "disadvantage", "drawback", "harm", "damage", "deficit"}),
    ("NN", {"problem", "issue", "trouble", "difficulty", "obstacle", "challenge", "barrier", "dilemma"}),
    ("NN", {"solution", "answer", "resolution", "remedy", "way", "fix"}),
    ("NN", {"rule", "regulation", "policy", "law", "guideline", "norm", "statute"}),
    ("NN", {"goal", "aim", "objective", "purpose", "target", "intention", "mission"}),
    ("NN", {"reason", "cause", "factor", "motive", "basis", "ground"}),
    ("NN", {"effect", "impact", "result", "consequence", "outcome", "influence", "aftermath"}),
    ("NN", {"waste", "garbage", "trash", "rubbish", "litter", "refuse", "debris", "pollution"}),
    ("NN", {"environment", "nature", "ecosystem", "habitat", "surroundings", "ecology"}),
    ("NN", {"plant", "vegetation", "flora", "crop", "greenery"}),
    ("NN", {"animal", "creature", "fauna", "wildlife", "beast"}),
    ("NN", {"human", "person", "people", "individual", "citizen", "inhabitant", "resident", "society", "community"}),
    ("NN", {"income", "revenue", "earnings", "salary", "wage", "pay", "proceeds"}),
    ("NN", {"cost", "expense", "price", "fee", "charge", "expenditure", "spending"}),
    ("NN", {"goods", "product", "commodity", "item", "merchandise", "ware"}),
    ("NN", {"buyer", "consumer", "customer", "client", "purchaser", "shopper", "user"}),
    ("NN", {"seller", "merchant", "vendor", "trader", "producer", "supplier", "retailer"}),
    ("NN", {"job", "work", "occupation", "profession", "career", "employment", "task", "duty"}),
    ("NN", {"aid", "assistance", "subsidy", "grant", "relief", "donation", "charity"}),
    ("NN", {"freedom", "liberty", "independence", "autonomy", "sovereignty"}),
    ("NN", {"war", "battle", "conflict", "combat", "struggle", "fight", "rebellion", "revolution"}),
    ("NN", {"agreement", "treaty", "pact", "accord", "deal", "contract", "consensus", "settlement"}),
    ("NN", {"leader", "ruler", "chief", "commander", "head", "president", "king", "director"}),
    ("NN", {"story", "tale", "narrative", "account", "chronicle", "history", "legend"}),
    ("NN", {"message", "moral", "lesson", "advice", "meaning", "point"}),
]


def _normalize_synset_entry(text: str) -> str:
    """Normalisasi kata/frasa synset menggunakan pipeline standar (slang + stem/lemma)."""
    tokens = [
        _normalize_word(t)
        for t in text.lower().strip().split()
        if t and t not in STOP_WORDS
    ]
    tokens = [t for t in tokens if t]
    if not tokens:
        # Jika semua token adalah stopword, gunakan _normalize_word langsung tanpa filter stopword
        tokens = [_normalize_word(t) for t in text.lower().strip().split() if t]
    return " ".join(tokens)


# Bangun indeks synset ternormalisasi saat modul dimuat:
# _NORMALIZED_CLUSTERS: List[Tuple[pos_cat, Set[normalized_str]]]
# _LEMMA_TO_CLUSTER_IDS: Dict[normalized_str, List[int]]
# _NORM_TO_DISPLAY: Dict[normalized_str, str]
_NORMALIZED_CLUSTERS: List[Tuple[str, Set[str]]] = []
_LEMMA_TO_CLUSTER_IDS: Dict[str, List[int]] = {}
_NORM_TO_DISPLAY: Dict[str, str] = {}


def _build_synset_index() -> None:
    """Membangun indeks pencarian cepat dari RAW_SYNSET_CLUSTERS."""
    if _NORMALIZED_CLUSTERS:
        return
    for pos_cat, raw_words in RAW_SYNSET_CLUSTERS:
        norm_set: Set[str] = set()
        for w in sorted(raw_words):
            nw = _normalize_synset_entry(w)
            # Jangan masukkan kata negasi, angka, atau kata kosong ke dalam synset
            if nw and nw not in NEGATION_WORDS and nw not in NUMBER_WORDS:
                norm_set.add(nw)
                if nw not in _NORM_TO_DISPLAY or len(w) < len(_NORM_TO_DISPLAY[nw]):
                    _NORM_TO_DISPLAY[nw] = w
        if len(norm_set) > 1:
            cid = len(_NORMALIZED_CLUSTERS)
            _NORMALIZED_CLUSTERS.append((pos_cat, norm_set))
            for nw in norm_set:
                _LEMMA_TO_CLUSTER_IDS.setdefault(nw, []).append(cid)


_build_synset_index()


def _is_guarded_token(token: str) -> bool:
    """
    Periksa apakah token tidak boleh diekspansi (kata negasi, angka, digit,
    stopword murni, atau kata ambigu/polisemi).
    """
    if not token or len(token) <= 2:
        return True
    if token.isdigit():
        return True
    if token in NEGATION_WORDS or token in NUMBER_WORDS or token in STOP_WORDS:
        return True
    if token in POLYSEMOUS_GUARD_WORDS:
        return True
    return False


def _pos_compatible(query_pos: Optional[str], cluster_pos: str) -> bool:
    """Periksa apakah POS tag kompatibel dengan kategori POS cluster."""
    if not query_pos or cluster_pos == "ANY" or query_pos == "UNK":
        return True
    if query_pos == cluster_pos:
        return True
    # Dalam Bahasa Indonesia/Inggris, nomina deverbal (NN) dan verba (VB) atau adjektiva (JJ)
    # setelah stemming Sastrawi sering memiliki bentuk dasar yang sama (misal: "cemar", "rusak", "untung").
    # Namun kita cegah pencampuran ekstrem jika POS sangat spesifik.
    return True


def _get_wordnet_synonyms(norm_token: str, pos_tag: Optional[str] = None) -> Set[str]:
    """
    Ambil sinonim dari NLTK WordNet (en & ind jika omw-1.4 tersedia) secara aman.
    Hanya mengembalikan sinonim 1-kata yang bukan antonim/negasi/polisemi.
    """
    results: Set[str] = set()
    if not _WORDNET_AVAILABLE or _is_guarded_token(norm_token):
        return results

    wn_pos = None
    if pos_tag == "VB":
        wn_pos = wn.VERB
    elif pos_tag == "JJ":
        wn_pos = wn.ADJ
    elif pos_tag == "NN":
        wn_pos = wn.NOUN

    try:
        # Cegah homograf kata dasar Bahasa Indonesia ("air", "ban", "cat", "jam")
        # tercemar sinonim WordNet Bahasa Inggris; gunakan korpus OMW 'ind' jika ID.
        if norm_token in _id_lemmatizer.root_word:
            if not _OMW_AVAILABLE:
                return results
            synsets = wn.synsets(norm_token, pos=wn_pos, lang="ind")[:2]
            for syn in synsets:
                for lemma_name in syn.lemma_names("ind"):
                    cand = lemma_name.lower().replace("_", " ")
                    if " " not in cand:
                        norm_cand = _normalize_word(cand, lang="id")
                        if not _is_guarded_token(norm_cand):
                            results.add(norm_cand)
            return results

        # Batasi hanya pada synset utama (maksimal 2 synset teratas) untuk mencegah
        # perluasan makna sekunder/idiomatik yang terlalu jauh (polysemy drift).
        synsets = wn.synsets(norm_token, pos=wn_pos)[:2]
        for syn in synsets:
            for lemma in syn.lemmas():
                cand = lemma.name().lower().replace("_", " ")
                if " " not in cand:
                    norm_cand = _normalize_word(cand, lang="en")
                    if not _is_guarded_token(norm_cand):
                        results.add(norm_cand)
            if _OMW_AVAILABLE:
                for lemma_name in syn.lemma_names("ind"):
                    cand = lemma_name.lower().replace("_", " ")
                    if " " not in cand:
                        norm_cand = _normalize_word(cand, lang="id")
                        if not _is_guarded_token(norm_cand):
                            results.add(norm_cand)
    except Exception:
        pass

    return results


def get_synonyms(word_or_phrase: str, pos_tag: Optional[str] = None) -> Set[str]:
    """
    Mengembalikan himpunan sinonim ternormalisasi (bentuk dasar/stem) untuk
    sebuah kata atau frasa yang sudah maupun belum dipraproses.

    Args:
        word_or_phrase: Kata atau frasa input.
        pos_tag: Opsional kelas kata ("VB", "JJ", "NN", dll.) dari `get_pos_tags`.

    Returns:
        Set[str] berisi bentuk ternormalisasi dari sinonim-sinonim yang valid
        (tidak termasuk kata input itu sendiri).
    """
    if not word_or_phrase or not word_or_phrase.strip():
        return set()

    norm_query = _normalize_synset_entry(word_or_phrase)
    if not norm_query:
        return set()

    # Jika berupa 1 kata dan termasuk kata yang dilindungi (negasi, angka, polisemi), jangan ekspansi
    if " " not in norm_query and _is_guarded_token(norm_query):
        return set()

    synonyms: Set[str] = set()

    # 1. Cari di Tesaurus Bilingual Terkurasi
    cluster_ids = _LEMMA_TO_CLUSTER_IDS.get(norm_query, [])
    for cid in cluster_ids:
        c_pos, c_set = _NORMALIZED_CLUSTERS[cid]
        if _pos_compatible(pos_tag, c_pos):
            for item in c_set:
                if item != norm_query and not _is_guarded_token(item):
                    synonyms.add(item)

    # 2. Cari di NLTK WordNet (jika 1 kata dan WordNet tersedia secara lokal)
    if " " not in norm_query and _WORDNET_AVAILABLE:
        wn_syns = _get_wordnet_synonyms(norm_query, pos_tag=pos_tag)
        for item in wn_syns:
            if item != norm_query and not _is_guarded_token(item):
                synonyms.add(item)

    return synonyms


def _preprocess_variant_with_stops(text: str, flags: Optional[Any] = None) -> str:
    """Memproses varian kata kunci tanpa menghapus stopwords agar frasa kunci tetap utuh."""
    if flags is None:
        return preprocess_text(text, remove_stopwords=False)
    if hasattr(flags, "model_dump"):
        d = flags.model_dump()
    elif isinstance(flags, dict):
        d = dict(flags)
    else:
        d = dict(getattr(flags, "__dict__", {}))
    d["remove_stopwords"] = False
    return preprocess_text(text, flags=d)


def expand_keyword_variants(
    variants: List[str],
    flags: Optional[Any] = None,
) -> List[Tuple[str, str]]:
    """
    Menghasilkan daftar pasangan `(processed_synonym, source_variant_text)` untuk
    daftar varian kata kunci non-strict pada Rule-Based.

    Untuk varian 1 kata: mengekspansi kata tersebut ke seluruh sinonimnya.
    Untuk varian frasa multi-kata (2-3 kata): jika salah satu kata inti memiliki
    sinonim di tesaurus, menghasilkan variasi frasa dengan kata sinonim tersebut
    sambil tetap menjaga posisi kata stopword di dalam frasa agar konteks tidak berubah.
    """
    expanded_pairs: List[Tuple[str, str]] = []
    seen_processed: Set[str] = set()

    # Kumpulkan bentuk processed dari varian asli agar tidak duplikat
    for v in variants:
        pv = _preprocess_variant_with_stops(v, flags=flags)
        if pv:
            seen_processed.add(pv)

    for v in variants:
        pv = _preprocess_variant_with_stops(v, flags=flags)
        if not pv:
            continue

        tokens = pv.split()
        pos_tags = get_pos_tags(v)
        pos_map = { _normalize_word(w): tag for w, tag in pos_tags }

        if len(tokens) == 1:
            tok = tokens[0]
            if _is_guarded_token(tok):
                continue
            syns = get_synonyms(tok, pos_tag=pos_map.get(tok))
            for syn in sorted(syns):
                if syn not in seen_processed:
                    seen_processed.add(syn)
                    expanded_pairs.append((syn, _NORM_TO_DISPLAY.get(syn, syn)))
        elif 2 <= len(tokens) <= 3:
            # Pertama, cek apakah frasa utuh ada di synset
            phrase_syns = get_synonyms(pv)
            for syn in sorted(phrase_syns):
                if syn not in seen_processed:
                    seen_processed.add(syn)
                    expanded_pairs.append((syn, _NORM_TO_DISPLAY.get(syn, syn)))

            # Kedua, substitusi 1 token inti dalam frasa dengan sinonimnya
            for idx, tok in enumerate(tokens):
                if _is_guarded_token(tok):
                    continue
                syns = get_synonyms(tok, pos_tag=pos_map.get(tok))
                for syn in sorted(syns):
                    if " " in syn:
                        continue
                    new_tokens = tokens[:idx] + [syn] + tokens[idx + 1 :]
                    new_phrase = " ".join(new_tokens)
                    if new_phrase not in seen_processed:
                        seen_processed.add(new_phrase)
                        expanded_pairs.append((new_phrase, new_phrase))

    return expanded_pairs


def align_tokens_for_lsa(processed_student: str, processed_ref: str) -> str:
    """
    Menyelaraskan token pada `processed_student` ke token kanonik yang ada pada
    `processed_ref` apabila keduanya berada dalam synset yang sama (Canonical Token Alignment).

    Aturan Keamanan (Safety Guardrails):
    - Token yang sudah ada secara eksak di `processed_ref` TIDAK diubah.
    - Token negasi (`NEGATION_WORDS`) dan kata bilangan (`NUMBER_WORDS`) TIDAK pernah diubah,
      sehingga penalti negasi (`_negation_penalty_factor`) dan cek angka tetap 100% akurat.
    - Hanya mengganti token siswa jika sinonimnya benar-benar muncul di `processed_ref`.
    """
    if not processed_student or not processed_ref:
        return processed_student

    ref_tokens_list = processed_ref.split()
    ref_tokens_set = set(ref_tokens_list)
    student_tokens = processed_student.split()

    # Buat pemetaan balik: untuk setiap token di ref_tokens, daftarkan sinonimnya -> ref_token
    # Jika ada lebih dari satu ref_token dalam synset yang sama, prioritaskan urutan kemunculan pertama di ref
    synonym_to_ref_canonical: Dict[str, str] = {}
    for r_tok in ref_tokens_list:
        if _is_guarded_token(r_tok):
            continue
        r_syns = get_synonyms(r_tok)
        for syn in r_syns:
            if syn not in ref_tokens_set and syn not in synonym_to_ref_canonical:
                synonym_to_ref_canonical[syn] = r_tok

    if not synonym_to_ref_canonical:
        return processed_student

    aligned_tokens: List[str] = []
    for s_tok in student_tokens:
        # Jika token siswa sudah ada di referensi, atau merupakan kata negasi/angka/guarded, biarkan utuh
        if s_tok in ref_tokens_set or _is_guarded_token(s_tok):
            aligned_tokens.append(s_tok)
        elif s_tok in synonym_to_ref_canonical:
            aligned_tokens.append(synonym_to_ref_canonical[s_tok])
        else:
            aligned_tokens.append(s_tok)

    return " ".join(aligned_tokens)

