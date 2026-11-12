# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field
from typing import List, Optional

class KeywordItem(BaseModel):
    keyword: str = Field(..., description="Kata kunci")
    weight: float = Field(1.0, ge=0.0, description="Bobot kata kunci, ditentukan guru (default 1.0)")
    strict: bool = Field(False, description="Jika True, maka dicocokkan menggunakan raw case-sensitive match")

class MathStepItem(BaseModel):
    equation: str = Field(..., description="Persamaan matematis yang diharapkan (contoh: 'Q = 10000 - 0.5*P')")
    weight: float = Field(1.0, ge=0.0, description="Bobot poin untuk langkah hitungan ini")

class PreprocessingFlags(BaseModel):
    clean_html: bool = Field(True, description="Bersihkan tag HTML dan normalisasi jeda baris")
    case_folding: bool = Field(True, description="Ubah seluruh karakter menjadi huruf kecil (lowercase)")
    expand_contractions: bool = Field(True, description="Ekspansi singkatan bahasa Inggris (don't -> do not)")
    convert_numbers: bool = Field(True, description="Konversi selektif bilangan 0-100 menjadi kata (num2words)")
    remove_punctuation: bool = Field(True, description="Hapus tanda baca")
    preserve_years: bool = Field(True, description="Pertahankan angka tahun 4 digit rentang 1000 - 2099")
    remove_stopwords: bool = Field(True, description="Hapus kata henti (stopwords)")
    use_lemmatization: bool = Field(True, description="Lemmatisasi bilingual (nlp-id untuk ID, spaCy untuk EN)")


class EssayRequest(BaseModel):
    student_answer: str = Field(..., description="Jawaban esai dari siswa (bisa campuran Bahasa Indonesia & Inggris)")
    reference_answers: List[str] = Field(
        ...,
        min_length=1,
        description="Daftar kunci jawaban dari guru (bisa lebih dari satu, misal versi Bahasa Indonesia dan versi Bahasa Inggris)"
    )
    keywords: List[KeywordItem] = Field(default_factory=list, description="Daftar kata kunci beserta bobot dinamisnya")
    required_keywords_count: Optional[int] = Field(
        None,
        ge=1,
        description="Jumlah target minimal kata kunci yang harus disebutkan siswa untuk mencapai nilai 100% pada rule-based. Jika None, seluruh kata kunci harus ada."
    )
    math_steps: List[MathStepItem] = Field(default_factory=list, description="Daftar ekspektasi langkah persamaan matematis")
    rule_weight: float = Field(0.5, ge=0.0, le=1.0, description="Bobot Rule-Based (0.0 - 1.0), ditentukan guru")
    lsa_weight: float = Field(0.5, ge=0.0, le=1.0, description="Bobot LSA (0.0 - 1.0), ditentukan guru")
    math_weight: float = Field(0.0, ge=0.0, le=1.0, description="Bobot Evaluasi Matematis (0.0 - 1.0)")
    tgate: float = Field(0.0, ge=0.0, le=100.0, description="Threshold Gate (0-100) untuk minimal skor faktual. LSA hanya akan dihitung jika skor faktual >= tgate.")
    
    # Feature Flags konfigurasi preprocessing (bisa dikirim nested atau langsung di root)
    preprocessing_flags: Optional[PreprocessingFlags] = Field(
        default=None,
        description="Objek konfigurasi On/Off fitur preprocessing untuk eksperimen akurasi"
    )
    clean_html: Optional[bool] = Field(None, description="Shortcut On/Off pembersihan HTML")
    case_folding: Optional[bool] = Field(None, description="Shortcut On/Off case folding")
    expand_contractions: Optional[bool] = Field(None, description="Shortcut On/Off ekspansi kontraksi")
    convert_numbers: Optional[bool] = Field(None, description="Shortcut On/Off num2words (default True)")
    remove_punctuation: Optional[bool] = Field(None, description="Shortcut On/Off pembersihan tanda baca")
    preserve_years: Optional[bool] = Field(None, description="Shortcut On/Off preservasi tahun")
    remove_stopwords: Optional[bool] = Field(None, description="Shortcut On/Off stopword removal")
    use_lemmatization: Optional[bool] = Field(None, description="Shortcut On/Off lemmatisasi")
    
    max_score: float = Field(100.0, ge=0.0, description="Skor maksimal/poin penuh untuk butir soal ini (default 100.0)")

    def get_preprocessing_flags(self) -> PreprocessingFlags:
        """Menggabungkan preprocessing_flags objek dengan shortcut flags (jika ada)."""
        base = self.preprocessing_flags or PreprocessingFlags()
        data = base.model_dump()
        for field in [
            "clean_html", "case_folding", "expand_contractions", "convert_numbers",
            "remove_punctuation", "preserve_years", "remove_stopwords", "use_lemmatization"
        ]:
            val = getattr(self, field, None)
            if val is not None:
                data[field] = val
        return PreprocessingFlags(**data)


class GraderResponse(BaseModel):
    final_score: float = Field(..., description="Skor akhir gabungan (0 - 100)")
    scaled_score: float = Field(..., description="Skor akhir yang disesuaikan dengan max_score soal")
    rule_based_score: float = Field(..., description="Skor dari pencocokan keyword berbobot (0 - 100)")
    lsa_score: float = Field(..., description="Skor terbaik dari LSA terhadap semua kunci jawaban (0 - 100)")
    math_score: float = Field(0.0, description="Skor evaluasi matematis menggunakan SymPy (0 - 100)")
    matched_keywords: List[str] = Field(..., description="Kata kunci yang berhasil dicocokkan")


class PosTagItem(BaseModel):
    token: str = Field(..., description="Token kata atau simbol")
    tag: str = Field(..., description="Tag Part-of-Speech")


class PosTagRequest(BaseModel):
    text: str = Field(..., description="Teks kalimat esai yang akan dianalisis kelas katanya")
    refine_foreign: bool = Field(False, description="Jika True, memperjelas token kata asing (FW) dengan tag POS bahasa Inggris")


class PosTagResponse(BaseModel):
    tags: List[PosTagItem] = Field(..., description="Daftar pasangan token dan label kelas katanya")


class LemmatizeRequest(BaseModel):
    text: str = Field(..., description="Teks kalimat esai yang akan di-lemmatize")


class LemmatizeResponse(BaseModel):
    original_text: str = Field(..., description="Teks masukan asli")
    lemmatized_text: str = Field(..., description="Teks setelah seluruh kata diubah ke bentuk lemma dasar")

