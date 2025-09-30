import os
import re
import fitz  # PyMuPDF
import pytesseract
import cv2
import numpy as np
from PIL import Image
import pandas as pd

# Tesseract 실행파일 경로 (Windows 설치 경로 확인 필수)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 바탕화면/target_folder
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
target_folder = os.path.join(desktop, "target_folder")

# PDF 파일 목록 가져오기
pdf_files = [f for f in os.listdir(target_folder) if f.lower().endswith(".pdf")]

results = []

for pdf_file in pdf_files:
    file_path = os.path.join(target_folder, pdf_file)
    print(f"=== {pdf_file} ===")

    doc = fitz.open(file_path)
    for page_num in range(len(doc)):
        page = doc[page_num]

        # PDF 페이지 → 이미지 변환
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # OpenCV 전처리 (그레이스케일 + Adaptive Threshold)
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 35, 11
        )
        pil_img = Image.fromarray(thresh)

        # OCR 실행 (한글 + 숫자, PSM 옵션)
        custom_config = "--psm 6 --oem 3"
        text = pytesseract.image_to_string(pil_img, lang="kor+eng", config=custom_config)

        # ===============================
        # 정규식으로 데이터 추출
        # ===============================

        # 거래일자 (YYYY-MM-DD HH:MM:SS)
        date_match = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", text)
        date = date_match.group(0) if date_match else None

        # 금액 (1,234원 같은 패턴)
        amount_match = re.search(r"(\d{1,3}(?:,\d{3})*)\s*원", text)
        amount = amount_match.group(1) if amount_match else None

        # 가맹점명 (단순히 "가맹점명" 이후 한 줄 추출)
        store = None
        lines = text.splitlines()
        for idx, line in enumerate(lines):
            if "가맹점" in line:
                if idx + 1 < len(lines):
                    store = lines[idx + 1].strip()
                break

        results.append({
            "파일명": pdf_file,
            "거래일자": date,
            "가맹점명": store,
            "금액": amount
        })

# DataFrame으로 정리
df = pd.DataFrame(results)
print(df)

# CSV 저장
output_path = os.path.join(desktop, "expense_result.csv")
df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"\nCSV 저장 완료 → {output_path}")
