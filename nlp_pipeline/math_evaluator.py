import re
import sympy
from typing import List, Optional
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from schemas import MathStepItem

def preprocess_math_text(text: str) -> List[str]:
    """
    Ekstrak semua baris/potongan teks yang mengandung persamaan (=).
    Mendukung => atau newline atau koma sebagai pemisah.
    """
    text = text.replace("=>", "\n")
    text = re.sub(r'[,;]\s+', '\n', text)
    lines = text.split("\n")
    
    equations = []
    for line in lines:
        if "=" in line:
            # Hapus kata-kata bahasa Indonesia (huruf >2 karakter)
            cleaned = re.sub(r'\b[a-zA-Z]{3,}\b', '', line)
            
            # Hapus karakter non-matematis
            cleaned = re.sub(r'[^a-zA-Z0-9\+\-\*\/\=\(\)\.\,]', ' ', cleaned)
            
            # Format angka (ribu vs desimal)
            cleaned = re.sub(r'(?<=\d)\.(?=\d{3}\b)', '', cleaned)
            cleaned = re.sub(r'(?<=\d)\,(?=\d)', '.', cleaned)
            
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            
            if "=" in cleaned:
                equations.append(cleaned)
                
    return equations

def parse_equation(eq_str: str) -> Optional[sympy.Expr]:
    """
    Mengurai string persamaan (LHS = RHS) menjadi ekspresi SymPy (LHS - RHS).
    """
    if "=" not in eq_str:
        return None
        
    left_str, right_str = eq_str.split("=", 1)
    
    transformations = (standard_transformations + (implicit_multiplication_application,))
    
    # Q adalah reserved word di sympy (sympy.Q = AssumptionKeys), jadi kita paksa jadi Symbol
    local_dict = {
        'Q': sympy.Symbol('Q'),
        'P': sympy.Symbol('P'),
        'Qd': sympy.Symbol('Qd'),
        'Qs': sympy.Symbol('Qs'),
        'Pd': sympy.Symbol('Pd'),
        'Ps': sympy.Symbol('Ps'),
        'S': sympy.Symbol('S'),
        'D': sympy.Symbol('D')
    }
    
    try:
        lhs = parse_expr(left_str, local_dict=local_dict, transformations=transformations)
        rhs = parse_expr(right_str, local_dict=local_dict, transformations=transformations)
        return lhs - rhs
    except Exception:
        return None

def calculate_math_score(student_answer: str, expected_steps: List[MathStepItem]) -> float:
    """
    Mengevaluasi jawaban matematika siswa terhadap langkah-langkah yang diharapkan.
    """
    if not expected_steps:
        return 0.0
        
    extracted_lines = preprocess_math_text(student_answer)
    
    student_exprs = []
    for line in extracted_lines:
        expr = parse_equation(line)
        if expr is not None:
            student_exprs.append(expr)
            
    total_weight = sum(step.weight for step in expected_steps)
    if total_weight == 0:
        return 0.0
        
    score = 0.0
    for step in expected_steps:
        # Preprocess expected equation just in case it has 10.000 or commas
        expected_cleaned = preprocess_math_text(step.equation)
        if not expected_cleaned:
            continue
            
        expected_expr = parse_equation(expected_cleaned[0])
        if expected_expr is None:
            continue
            
        step_matched = False
        for s_expr in student_exprs:
            try:
                # Cek ekuivalensi (A = B sama dengan A - B = 0 atau B - A = 0)
                diff = sympy.simplify(s_expr - expected_expr)
                if diff == 0:
                    step_matched = True
                    break
                    
                diff_reverse = sympy.simplify(s_expr + expected_expr) # Jika ruas tertukar tanda
                if diff_reverse == 0:
                    step_matched = True
                    break
            except Exception:
                continue
                
        if step_matched:
            score += step.weight
            
    return (score / total_weight) * 100.0
