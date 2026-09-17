from schemas import EssayRequest
from grader import grade_essay

req = EssayRequest(
    student_answer="Orator hendak menegaskan kalau kebersihan lingkungan sekolah (terutama kelas dan halaman sekolah) adalah kewajiban yang dibebankan pada seluruh warga sekolah.",
    reference_answers=["Pesan yang ingin disampaikan: Semua warga sekolah harus menjaga kebersihan lingkungan sekolah bersama-sama."],
    keywords=[
        {"keyword": "warga sekolah", "weight": 1.0},
        {"keyword": "kebersihan lingkungan", "weight": 1.0}
    ],
    rule_weight=0.0,
    lsa_weight=1.0
)

res = grade_essay(req)
print(f"LSA Score: {res.lsa_score}")
