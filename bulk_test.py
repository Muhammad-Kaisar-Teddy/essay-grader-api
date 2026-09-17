import argparse
import os
from typing import List
import pandas as pd
from schemas import EssayRequest, KeywordItem
from grader import grade_essay

def parse_reference_answers(text: str) -> List[str]:
    """Parse reference answers dipisahkan dengan karakter '|'"""
    if not isinstance(text, str):
        return []
    return [ans.strip() for ans in text.split('|') if ans.strip()]

def parse_keywords(text: str) -> List[KeywordItem]:
    """
    Parse keywords dari format string.
    Contoh: "tumbuhan:1.5; **Jakarta:2.0; **sungai, pasar:1.0"
    Gunakan titik koma (;) sebagai pemisah antar keyword.
    Jika diawali dengan '**', set strict=True.
    """
    keywords_list = []
    if not isinstance(text, str):
        return keywords_list
        
    items = [k.strip() for k in text.split(';') if k.strip()]
    for item in items:
        # Cek strict mode
        is_strict = False
        if item.startswith('**'):
            is_strict = True
            item = item[2:].strip()  # Hapus **

        if ':' in item:
            parts = item.split(':')
            keyword = parts[0].strip()
            try:
                weight = float(parts[1].strip())
            except ValueError:
                weight = 1.0
        else:
            keyword = item
            weight = 1.0
        
        keywords_list.append(KeywordItem(keyword=keyword, weight=weight, strict=is_strict))
    
    return keywords_list

def main():
    parser = argparse.ArgumentParser(description="Uji coba AEG Model menggunakan file CSV atau Excel (.xlsx)")
    parser.add_argument("input_file", help="Path ke file input (CSV atau XLSX)")
    parser.add_argument("output_file", help="Path ke file output (akan mengikuti ekstensi input, disarankan .csv atau .xlsx)")
    args = parser.parse_args()

    # 1. Baca data berdasarkan ekstensi file
    input_ext = os.path.splitext(args.input_file)[1].lower()
    try:
        if input_ext == '.csv':
            df = pd.read_csv(args.input_file)
        elif input_ext in ['.xls', '.xlsx']:
            df = pd.read_excel(args.input_file)
        else:
            print(f"Error: Format file {input_ext} tidak didukung. Gunakan .csv atau .xlsx")
            return
    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' tidak ditemukan!")
        return

    # Validasi keberadaan kolom
    required_cols = ['student_answer', 'reference_answers', 'keywords']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Kolom wajib '{col}' tidak ditemukan di dalam file!")
            return

    results = []

    # 2. Proses tiap baris
    print(f"Memproses {len(df)} data jawaban esai...")
    for idx, row in df.iterrows():
        try:
            # Gunakan nilai default jika kolom bobot tidak ada atau kosong.
            # Handle juga format koma desimal Indonesia (misal '0,9' -> '0.9')
            rule_w = float(str(row['rule_weight']).replace(',', '.')) if 'rule_weight' in row and pd.notna(row['rule_weight']) else 0.5
            lsa_w = float(str(row['lsa_weight']).replace(',', '.')) if 'lsa_weight' in row and pd.notna(row['lsa_weight']) else 0.5
            
            # Ambil max_score (default 100.0)
            max_s = float(str(row['max_score']).replace(',', '.')) if 'max_score' in row and pd.notna(row['max_score']) else 100.0
            
            req = EssayRequest(
                student_answer=str(row['student_answer']),
                reference_answers=parse_reference_answers(str(row['reference_answers'])),
                keywords=parse_keywords(str(row['keywords'])),
                rule_weight=rule_w,
                lsa_weight=lsa_w,
                max_score=max_s
            )
            
            resp = grade_essay(req)
            
            row_result = row.to_dict()
            row_result['final_score'] = resp.final_score
            row_result['scaled_score'] = resp.scaled_score
            row_result['rule_based_score'] = resp.rule_based_score
            row_result['lsa_score'] = resp.lsa_score
            row_result['matched_keywords'] = " | ".join(resp.matched_keywords)
            results.append(row_result)
            print(f"  [{idx+1}/{len(df)}] Score = {resp.final_score}")
            
        except Exception as e:
            print(f"  [{idx+1}/{len(df)}] ERROR: {e}")
            row_dict = row.to_dict()
            row_dict.update({
                "final_score": "ERROR", 
                "rule_based_score": "", 
                "lsa_score": "", 
                "matched_keywords": str(e)
            })
            results.append(row_dict)

    # 3. Tulis hasil
    out_df = pd.DataFrame(results)
    out_ext = os.path.splitext(args.output_file)[1].lower()
    
    if out_ext == '.csv':
        out_df.to_csv(args.output_file, index=False)
    elif out_ext in ['.xls', '.xlsx']:
        out_df.to_excel(args.output_file, index=False)
    else:
        # Default ke CSV jika pengguna tidak memberi ekstensi
        out_df.to_csv(args.output_file + '.csv', index=False)
        args.output_file += '.csv'

    print(f"\nSelesai! Hasil grading disimpan di: {args.output_file}")

if __name__ == "__main__":
    main()
