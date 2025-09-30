import pytesseract
from pdf2image import convert_from_path
import pandas as pd

# Tesseract 실행파일 경로
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

file_path = r"C:\Users\user\Desktop\target_folder\1.pdf"
poppler_path = r"C:\Program Files\poppler-25.07.0\Library\bin"

# PDF → 이미지 변환
pages = convert_from_path(file_path, dpi=300, poppler_path=poppler_path)

all_dfs = []

for i, page in enumerate(pages, 1):
    print(f"Processing page {i}...")

    # OCR with 위치정보
    data = pytesseract.image_to_data(page, lang="kor+eng", output_type=pytesseract.Output.DATAFRAME)
    data["page"] = i  # 페이지 번호 추가
    all_dfs.append(data)

# 전체 데이터 합치기
df = pd.concat(all_dfs, ignore_index=True)

# CSV 저장
df.to_csv("ocr_with_positions.csv", index=False, encoding="utf-8-sig")
print("ocr_with_positions.csv 저장 완료!")
