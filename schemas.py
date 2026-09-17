# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field
from typing import List

class KeywordItem(BaseModel):
    keyword: str = Field(..., description="Kata kunci")
    weight: float = Field(1.0, ge=0.0, description="Bobot kata kunci, ditentukan guru (default 1.0)")
    strict: bool = Field(False, description="Jika True, maka dicocokkan menggunakan raw case-sensitive match")

class MathStepItem(BaseModel):
    equation: str = Field(..., description="Persamaan matematis yang diharapkan (contoh: 'Q = 10000 - 0.5*P')")
    weight: float = Field(1.0, ge=0.0, description="Bobot poin untuk langkah hitungan ini")

class EssayRequest(BaseModel):
    student_answer: str = Field(..., description="Jawaban esai dari siswa (bisa campuran Bahasa Indonesia & Inggris)")
    reference_answers: List[str] = Field(
        ...,
        min_length=1,
        description="Daftar kunci jawaban dari guru (bisa lebih dari satu, misal versi Bahasa Indonesia dan versi Bahasa Inggris)"
    )
    keywords: List[KeywordItem] = Field(default_factory=list, description="Daftar kata kunci beserta bobot dinamisnya")
    math_steps: List[MathStepItem] = Field(default_factory=list, description="Daftar ekspektasi langkah persamaan matematis")
    rule_weight: float = Field(0.5, ge=0.0, le=1.0, description="Bobot Rule-Based (0.0 - 1.0), ditentukan guru")
    lsa_weight: float = Field(0.5, ge=0.0, le=1.0, description="Bobot LSA (0.0 - 1.0), ditentukan guru")
    math_weight: float = Field(0.0, ge=0.0, le=1.0, description="Bobot Evaluasi Matematis (0.0 - 1.0)")
    max_score: float = Field(100.0, ge=0.0, description="Skor maksimal/poin penuh untuk butir soal ini (default 100.0)")

class GraderResponse(BaseModel):
    final_score: float = Field(..., description="Skor akhir gabungan (0 - 100)")
    scaled_score: float = Field(..., description="Skor akhir yang disesuaikan dengan max_score soal")
    rule_based_score: float = Field(..., description="Skor dari pencocokan keyword berbobot (0 - 100)")
    lsa_score: float = Field(..., description="Skor terbaik dari LSA terhadap semua kunci jawaban (0 - 100)")
    math_score: float = Field(0.0, description="Skor evaluasi matematis menggunakan SymPy (0 - 100)")
    matched_keywords: List[str] = Field(..., description="Kata kunci yang berhasil dicocokkan")
