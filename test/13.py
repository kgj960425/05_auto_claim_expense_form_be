import pytesseract
from pdf2image import convert_from_path
import tkinter as tk
from tkinter import filedialog, messagebox
import os
from datetime import datetime
import uuid

# --- OCR 경로 설정 (윈도우라면 Tesseract 설치 경로 확인 필요) ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 파일을 api 화 하기 전에 파일 대용량으로 올려서 처리하는 로직 생성하기 위해서 테스트용으로 올림.
def run_ocr_on_pdfs():
    # === 1️⃣ 파일 탐색기 열기 ===
    root = tk.Tk()
    root.withdraw()
    pdf_paths = filedialog.askopenfilenames(
        title="OCR 처리할 PDF 파일 선택 (복수 가능)",
        filetypes=[("PDF Files", "*.pdf")]
    )

    if not pdf_paths:
        messagebox.showinfo("알림", "선택된 파일이 없습니다.")
        return

    # === 2️⃣ 결과 폴더 생성 ===
    # 작업 시작 시간 기록
    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d%H%M%S")
    job_uuid = str(uuid.uuid4())[:8]  # UUID의 앞 8자리만 사용

    # 출력 폴더명: static/temp/작업시작시간_UUID
    output_folder = f"static/temp/{timestamp}_{job_uuid}"
    os.makedirs(output_folder, exist_ok=True)

    print(f"작업 시작: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"출력 폴더: {output_folder}")

    # === 3️⃣ 각 PDF 처리 ===
    for pdf_path in pdf_paths:
        filename = os.path.basename(pdf_path)
        print(f"[처리 중] {filename}")


if __name__ == "__main__":
    run_ocr_on_pdfs()
