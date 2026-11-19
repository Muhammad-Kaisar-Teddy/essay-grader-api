# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Literal, Union

LINGUISTIC_SUBJECT_KEYWORDS = (
    "bahasa", "indonesia", "inggris", "english", "sastra", "bindo", "bing",
    "arab", "jepang", "jerman", "mandarin", "prancis", "perancis", "daerah",
    "sunda", "jawa", "lampung", "linguistic", "linguistik",
)

EXACT_SUBJECT_KEYWORDS = (
    "matematika", "math", "fisika", "physics", "kimia", "chemistry",
    "akuntansi", "statistika", "exact", "eksakta",
)


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
    use_synonyms: bool = Field(True, description="Ekspansi sinonim otomatis (WordNet/Tesaurus) pada Rule-Based dan penyelarasan kanonik pada LSA")


class EssayRequest(BaseModel):
    subject: Optional[str] = Field(
        None,
        description="Nama mata pelajaran dari Sistem Informasi Asesmen (misal: 'Bahasa Indonesia (XI)', 'Matematika (X)', 'Informatika (XI)', 'Sejarah')"
    )
    essay_mode: Optional[Literal["linguistic", "exact", "conceptual"]] = Field(
        None,
        description="Mode esai eksplisit ('linguistic', 'exact', atau 'conceptual'). Jika kosong, otomatis ditentukan dari subject dan math_steps."
    )
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
    use_synonyms: Optional[bool] = Field(None, description="Shortcut On/Off ekspansi sinonim otomatis")
    
    max_score: float = Field(100.0, ge=0.0, description="Skor maksimal/poin penuh untuk butir soal ini (default 100.0)")
    
    @model_validator(mode="after")
    def _weights_sum_to_one(self):
        total = self.rule_weight + self.lsa_weight + self.math_weight
        if total > 0 and abs(total - 1.0) > 1e-6:
            raise ValueError(f"Total bobot rule, lsa, dan math harus = 1.0. Saat ini: {total}")
        return self

    def resolve_essay_mode(self) -> Literal["linguistic", "exact", "conceptual"]:
        """
        Menentukan mode esai ('linguistic', 'exact', 'conceptual') secara otomatis:
        1. Prioritas 1: Jika `essay_mode` ditentukan secara eksplisit.
        2. Prioritas 2: Jika soal memiliki `math_steps` atau `math_weight > 0` -> 'exact'.
        3. Prioritas 3: Berdasarkan pencocokan kata kunci pada nama `subject` dari Sistem Informasi Asesmen:
           - Mengandung kata kunci bahasa/sastra -> 'linguistic'
           - Mengandung kata kunci eksakta/matematika -> 'exact'
           - Selain itu (Sejarah, Informatika, Biologi, Ekonomi teori, dll.) -> 'conceptual'
        """
        if self.essay_mode in ("linguistic", "exact", "conceptual"):
            return self.essay_mode

        is_math_question = bool(self.math_steps) or (self.math_weight > 0)
        if is_math_question:
            return "exact"

        if self.subject and self.subject.strip():
            subj_norm = self.subject.strip().lower().replace("_", " ").replace("-", " ")
            if any(kw in subj_norm for kw in LINGUISTIC_SUBJECT_KEYWORDS):
                return "linguistic"
            if any(kw in subj_norm for kw in EXACT_SUBJECT_KEYWORDS):
                return "exact"
            return "conceptual"

        return "conceptual"

    def get_preprocessing_flags(self) -> PreprocessingFlags:
        """
        Menggabungkan preset dari mode esai ('linguistic', 'exact', 'conceptual')
        dengan `preprocessing_flags` objek dan shortcut flags eksplisit (jika ada).
        """
        base = self.preprocessing_flags or PreprocessingFlags()
        data = base.model_dump()

        mode = self.resolve_essay_mode()
        if self.preprocessing_flags is None:
            if mode == "exact":
                data["remove_punctuation"] = False
                data["convert_numbers"] = False
                data["preserve_years"] = False
                data["use_lemmatization"] = False
                data["use_synonyms"] = False
            elif mode in ("linguistic", "conceptual"):
                data["clean_html"] = True
                data["case_folding"] = True
                data["expand_contractions"] = True
                data["convert_numbers"] = True
                data["remove_punctuation"] = True
                data["preserve_years"] = True
                data["remove_stopwords"] = True
                data["use_lemmatization"] = True
                data["use_synonyms"] = True

        for field in [
            "clean_html", "case_folding", "expand_contractions", "convert_numbers",
            "remove_punctuation", "preserve_years", "remove_stopwords", "use_lemmatization",
            "use_synonyms"
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
    essay_mode: Optional[str] = Field(None, description="Mode esai yang aktif saat penilaian (linguistic / exact / conceptual)")


class BatchEssayItem(EssayRequest):
    id: Optional[Union[str, int]] = Field(
        None,
        description="ID unik butir soal/jawaban (misal id_detail atau id_jawaban dari CBT) untuk pemetaan hasil batch"
    )
    id_soal: Optional[Union[str, int]] = Field(
        None,
        description="ID butir soal untuk pengelompokan jawaban seluruh siswa ke dalam satu korpus LSA per soal"
    )
    student_id: Optional[Union[str, int]] = Field(
        None,
        description="ID unik siswa (id_siswa) pemilik jawaban"
    )
    student_name: Optional[str] = Field(
        None,
        description="Nama siswa pemilik jawaban (opsional)"
    )


class StudentAnswerInput(BaseModel):
    id: Optional[Union[str, int]] = Field(
        None,
        description="ID unik baris jawaban di database (misal id_jawaban / id_detail)"
    )
    student_id: Optional[Union[str, int]] = Field(
        None,
        description="ID unik siswa (id_siswa)"
    )
    student_name: Optional[str] = Field(
        None,
        description="Nama lengkap siswa (opsional)"
    )
    id_soal: Optional[Union[str, int]] = Field(
        None,
        description="ID butir soal yang dijawab oleh siswa"
    )
    student_answer: str = Field(
        ...,
        description="Teks jawaban esai dari siswa"
    )


class QuestionConfigInput(BaseModel):
    id_soal: Union[str, int] = Field(
        ...,
        description="ID unik butir soal (id_soal / id_detail)"
    )
    subject: Optional[str] = Field(
        None,
        description="Nama mata pelajaran"
    )
    essay_mode: Optional[Literal["linguistic", "exact", "conceptual"]] = Field(
        None,
        description="Mode esai eksplisit ('linguistic', 'exact', atau 'conceptual')"
    )
    reference_answers: List[str] = Field(
        ...,
        min_length=1,
        description="Daftar kunci jawaban dari guru"
    )
    keywords: List[KeywordItem] = Field(
        default_factory=list,
        description="Daftar kata kunci beserta bobot dinamisnya"
    )
    required_keywords_count: Optional[int] = Field(
        None,
        ge=1,
        description="Jumlah target minimal kata kunci untuk nilai 100% pada rule-based"
    )
    math_steps: List[MathStepItem] = Field(
        default_factory=list,
        description="Daftar ekspektasi langkah persamaan matematis"
    )
    rule_weight: float = Field(0.5, ge=0.0, le=1.0, description="Bobot Rule-Based (0.0 - 1.0)")
    lsa_weight: float = Field(0.5, ge=0.0, le=1.0, description="Bobot LSA (0.0 - 1.0)")
    math_weight: float = Field(0.0, ge=0.0, le=1.0, description="Bobot Evaluasi Matematis (0.0 - 1.0)")
    tgate: float = Field(0.0, ge=0.0, le=100.0, description="Threshold Gate (0-100) untuk minimal skor faktual")
    max_score: float = Field(100.0, ge=0.0, description="Skor maksimal/poin penuh untuk butir soal ini")
    preprocessing_flags: Optional[PreprocessingFlags] = Field(default=None)
    clean_html: Optional[bool] = Field(None)
    case_folding: Optional[bool] = Field(None)
    expand_contractions: Optional[bool] = Field(None)
    convert_numbers: Optional[bool] = Field(None)
    remove_punctuation: Optional[bool] = Field(None)
    preserve_years: Optional[bool] = Field(None)
    remove_stopwords: Optional[bool] = Field(None)
    use_lemmatization: Optional[bool] = Field(None)
    use_synonyms: Optional[bool] = Field(None)
    students: Optional[List[StudentAnswerInput]] = Field(
        default=None,
        description="Opsional: Daftar jawaban seluruh siswa untuk butir soal ini"
    )


class BatchGraderResultItem(GraderResponse):
    id: Optional[Union[str, int]] = Field(
        None,
        description="ID unik butir soal/jawaban yang sesuai dengan item pada request batch"
    )
    id_soal: Optional[Union[str, int]] = Field(
        None,
        description="ID unik butir soal yang dinilai"
    )
    student_id: Optional[Union[str, int]] = Field(
        None,
        description="ID unik siswa yang dinilai"
    )
    student_name: Optional[str] = Field(
        None,
        description="Nama siswa yang dinilai (jika dikirim pada request)"
    )


class BatchEssayRequest(BaseModel):
    subject: Optional[str] = Field(
        None,
        description="Default nama mata pelajaran untuk seluruh item dalam batch jika item tidak menyertakan subject tersendiri"
    )
    essay_mode: Optional[Literal["linguistic", "exact", "conceptual"]] = Field(
        None,
        description="Default mode esai untuk seluruh item dalam batch jika item tidak menyertakan essay_mode tersendiri"
    )
    items: Optional[List[BatchEssayItem]] = Field(
        default=None,
        description="Daftar jawaban esai siswa (format flat) yang akan dikelompokkan berdasarkan id_soal dan dinilai sekaligus"
    )
    questions: Optional[List[QuestionConfigInput]] = Field(
        default=None,
        description="Daftar data soal (kunci jawaban, kata kunci, bobot, tgate, max_score) yang dipetakan berdasarkan id_soal"
    )
    answers: Optional[List[StudentAnswerInput]] = Field(
        default=None,
        description="Daftar seluruh jawaban siswa (student_id, student_name, id_soal, student_answer) untuk dipasangkan dengan questions"
    )

    @model_validator(mode="after")
    def _validate_batch_payload(self):
        has_items = bool(self.items)
        has_questions = bool(self.questions)
        if not has_items and not has_questions:
            raise ValueError("BatchEssayRequest harus berisi minimal 'items' atau 'questions'.")
        return self

    def to_normalized_items(self) -> List[BatchEssayItem]:
        """
        Menormalisasi berbagai bentuk payload batch (flat `items`, `questions` + `answers`,
        maupun `questions` dengan nested `students`) menjadi daftar `BatchEssayItem` yang seragam.
        """
        normalized: List[BatchEssayItem] = []

        # 1. Jika dikirim dalam bentuk flat items
        if self.items:
            for item in self.items:
                updates = {}
                if item.subject is None and self.subject is not None:
                    updates["subject"] = self.subject
                if item.essay_mode is None and self.essay_mode is not None:
                    updates["essay_mode"] = self.essay_mode
                normalized.append(item.model_copy(update=updates) if updates else item)

        # 2. Jika dikirim dalam bentuk terstruktur (questions + answers / nested students)
        if self.questions:
            question_map = {str(q.id_soal): q for q in self.questions}

            # 2a. Nested students di dalam masing-masing question
            for q in self.questions:
                if q.students:
                    q_data = q.model_dump(exclude={"students", "id_soal"})
                    if q_data.get("subject") is None and self.subject is not None:
                        q_data["subject"] = self.subject
                    if q_data.get("essay_mode") is None and self.essay_mode is not None:
                        q_data["essay_mode"] = self.essay_mode

                    for stu in q.students:
                        normalized.append(
                            BatchEssayItem(
                                id=stu.id,
                                id_soal=q.id_soal,
                                student_id=stu.student_id,
                                student_name=stu.student_name,
                                student_answer=stu.student_answer,
                                **q_data,
                            )
                        )

            # 2b. Daftar answers terpisah di level root yang merujuk ke id_soal
            if self.answers:
                for ans in self.answers:
                    if ans.id_soal is None or str(ans.id_soal) not in question_map:
                        raise ValueError(
                            f"Jawaban siswa (student_id={ans.student_id}) merujuk ke id_soal='{ans.id_soal}' "
                            f"yang tidak ditemukan pada daftar questions."
                        )
                    q = question_map[str(ans.id_soal)]
                    q_data = q.model_dump(exclude={"students", "id_soal"})
                    if q_data.get("subject") is None and self.subject is not None:
                        q_data["subject"] = self.subject
                    if q_data.get("essay_mode") is None and self.essay_mode is not None:
                        q_data["essay_mode"] = self.essay_mode

                    normalized.append(
                        BatchEssayItem(
                            id=ans.id,
                            id_soal=q.id_soal,
                            student_id=ans.student_id,
                            student_name=ans.student_name,
                            student_answer=ans.student_answer,
                            **q_data,
                        )
                    )

        if not normalized:
            raise ValueError("Tidak ada item jawaban siswa yang ditemukan untuk dinilai.")

        return normalized


class BatchGraderResponse(BaseModel):
    results: List[BatchGraderResultItem] = Field(
        ...,
        description="Daftar hasil penilaian untuk masing-masing item sesuai urutan request"
    )
    total_items: int = Field(..., description="Jumlah total butir esai yang dinilai dalam batch")


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

