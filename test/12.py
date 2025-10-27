import pymupdf4llm
import fitz
import pytesseract
import cv2
import numpy as np
from PIL import Image
import re
import os
import glob
from datetime import datetime
import uuid

FOLDER_PATH = r"C:\Users\user\Desktop\expense_cliaim"
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 작업 시작 시간 기록
start_time = datetime.now()
timestamp = start_time.strftime("%Y%m%d%H%M%S")
job_uuid = str(uuid.uuid4())[:8]  # UUID의 앞 8자리만 사용

# 출력 폴더명: 작업시작시간 + UUID
output_folder = f"extracted_images/{timestamp}_{job_uuid}"
os.makedirs(output_folder, exist_ok=True)


# 1. 이 uuid는 함수로 설정시 parameter로 받을 수 있게 할 것. 왜냐면 이 폴더 이름을 전 단계에서 알아야 다른 작업에서 이 폴더 지정해서 할테니까.
# 2. parameter로 받을거는. 작업폴더명, 작업자 아이디, 


print("PyMuPDF4LLM + OCR Pipeline (Batch Processing)")
print(f"Folder: {FOLDER_PATH}")
print(f"작업 시작: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"출력 폴더: {output_folder}")

# 폴더 내 모든 PDF 파일 찾기
pdf_files = glob.glob(os.path.join(FOLDER_PATH, "*.pdf"))
print(f"\n발견된 PDF 파일: {len(pdf_files)}개")

if not pdf_files:
    print("⚠️ PDF 파일이 없습니다!")
    exit()

for pdf_idx, PDF_PATH in enumerate(pdf_files, 1):
    print(f"\n{'='*80}")
    print(f"[{pdf_idx}/{len(pdf_files)}] 처리 중: {os.path.basename(PDF_PATH)}")
    print(f"{'='*80}")

    # 먼저 PyMuPDF로 텍스트가 있는지 확인
    doc = fitz.open(PDF_PATH)
    print(f"\nTotal pages: {len(doc)}")

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        print(f"\nPage {page_num + 1} - Text length: {len(text)}")
        if text.strip():
            print(f"First 200 chars: {text[:200]}")
        else:
            print("⚠️ No text found - This is likely an image-based PDF")

    doc.close()

    # PyMuPDF4LLM 시도 (다양한 옵션)
    print("\n=== PyMuPDF4LLM Extraction ===")

    # 옵션 1: 기본
    print("\n1. Basic extraction:")
    md_text = pymupdf4llm.to_markdown(PDF_PATH)
    print(f"   Length: {len(md_text)}")

    # 옵션 2: page_chunks 활성화
    print("\n2. With page_chunks:")
    md_chunks = pymupdf4llm.to_markdown(PDF_PATH, page_chunks=True)
    print(f"   Chunks: {len(md_chunks) if isinstance(md_chunks, list) else 'N/A'}")
    if isinstance(md_chunks, list) and md_chunks:
        print(f"   First chunk length: {len(md_chunks[0].get('text', ''))}")
        print(f"   First chunk preview: {md_chunks[0].get('text', '')[:300]}")

    # PDF별 임시 이미지 디렉토리 생성
    pdf_name = os.path.splitext(os.path.basename(PDF_PATH))[0]
    temp_image_dir = f"temp_extracted/{pdf_name}"
    os.makedirs(temp_image_dir, exist_ok=True)

    # 옵션 3: 이미지 포함
    print("\n3. With images:")
    try:
        md_with_images = pymupdf4llm.to_markdown(
            PDF_PATH,
            write_images=True,
            image_path=temp_image_dir,
            page_chunks=True
        )
        print(f"   Success! Chunks: {len(md_with_images) if isinstance(md_with_images, list) else 'N/A'}")
        if isinstance(md_with_images, list) and md_with_images:
            print(f"   First chunk: {md_with_images[0].get('text', '')[:300]}")
    except Exception as e:
        print(f"   Error: {e}")

    # ==========================
    # 4️⃣ 추출된 이미지에 OCR 적용
    # ==========================
    print("\n=== OCR on Extracted Images ===")

    if os.path.exists(temp_image_dir):
        image_files = [f for f in os.listdir(temp_image_dir) if f.endswith('.png')]
        print(f"Found {len(image_files)} images")

        for img_file in image_files:
            img_path = os.path.join(temp_image_dir, img_file)
            print(f"\n[{img_file}] OCR 시작...")

            # 이미지 로드
            img = Image.open(img_path)

            # Tesseract OCR
            ocr_text = pytesseract.image_to_string(img, lang='kor+eng', config='--psm 6')
            print(f"  텍스트 길이: {len(ocr_text)}")

            # 위치 정보 포함 OCR
            data = pytesseract.image_to_data(img, lang='kor+eng', config='--psm 6', output_type=pytesseract.Output.DICT)

            # 민감 정보 찾기: 카드번호 + 8~10자리 모든 숫자
            n_boxes = len(data['text'])
            positions = []

            print(f"  총 {n_boxes}개 텍스트 박스 검사 중...")

            for i in range(n_boxes):
                if int(data['conf'][i]) < 30:
                    continue

                text = data['text'][i].strip()
                if not text:
                    continue

                text_digits = re.sub(r'[^0-9*]', '', text)

                # 카드번호 (15자리 이상, *포함)
                if len(text_digits) >= 15:
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    positions.append(('카드번호', x, y, w, h))
                    print(f"  🔍 카드번호: {text} (위치: {x}, {y})")

                # 8~10자리 숫자 (날짜 제외)
                elif 8 <= len(text_digits) <= 10:
                    # 날짜 패턴 제외 (YYYY-MM-DD 형태)
                    if not re.match(r'20\d{2}[-/]\d{2}[-/]\d{2}', text):
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        positions.append((f'{len(text_digits)}자리', x, y, w, h))
                        print(f"  🔍 {len(text_digits)}자리 숫자: {text} (위치: {x}, {y})")

            # 블러 처리
            if positions:
                print(f"\n  블러 처리 시작... ({len(positions)}개 위치)")
                img_array = np.array(img)
                blurred = img_array.copy()

                for label, x, y, w, h in positions:
                    padding = 10
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(blurred.shape[1], x + w + padding)
                    y2 = min(blurred.shape[0], y + h + padding)

                    roi = blurred[y1:y2, x1:x2]
                    if roi.size > 0:
                        blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
                        blurred[y1:y2, x1:x2] = blurred_roi
                        print(f"  ✅ {label} 블러 완료")

                # 저장 - 파일명: 작업시작시간 + 원본파일명
                output_filename = f"{timestamp}_{pdf_name}_{img_file}"
                output_path = os.path.join(output_folder, output_filename)
                blurred_img = Image.fromarray(blurred)
                blurred_img.save(output_path)
                print(f"  💾 저장: {output_path}")

            print(f"\n  === OCR 텍스트 미리보기 ===")
            print(ocr_text[:500])
    else:
        print("⚠️ No images extracted")

# 임시 폴더 정리
import shutil
if os.path.exists("temp_extracted"):
    shutil.rmtree("temp_extracted")
    print("\n🗑️ 임시 파일 정리 완료")

print(f"\n{'='*80}")
print(f"✅ 전체 처리 완료! 총 {len(pdf_files)}개 PDF 파일 처리됨")
print(f"📁 결과 저장 위치: {output_folder}")
print(f"{'='*80}")
