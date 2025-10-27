from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
import tempfile, os
import cv2
import numpy as np
import fitz
import tempfile # 임시로 파일 생성. import 받은 파일이 with 구문 밖으로 나가자 마자 삭제 되도록
import base64
import requests
import pymupdf4llm
import pytesseract
from PIL import Image
import re
import pikepdf
from datetime import datetime
import uuid
import glob

from fastapi.responses import JSONResponse
from utils.crop_receipt import auto_crop_receipt

# Tesseract 경로 설정 (Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = FastAPI()

# Request Body 모델 정의
class MergeRequest(BaseModel):
    pdf_list: list[str]
    pdf_path: str
    output_file_name: str

@app.get("/serverCheck")
def custom_json():
    data = {"msg": "ok"}
    return JSONResponse(content=data, status_code=200)

@app.post("/ocr/receipt")
async def process_receipt(file: UploadFile = File(...), user_id: str = "tester"):
    # 1️⃣ PDF 임시 저장
    temp_pdf = tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") # 파일 닫히는 순간 od가 자동으로 삭제
    temp_pdf.write(await file.read())
    temp_pdf.close()

    # 2️⃣ 첫 페이지를 이미지로 변환 (PyMuPDF)
    pdf_doc = fitz.open(temp_pdf.name)
    page = pdf_doc.load_page(0)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img_path = temp_pdf.name.replace(".pdf", ".png")
    pix.save(img_path)

    # 3️⃣ 자동 crop
    cropped_img = auto_crop_receipt(img_path, "cropped.png")

    # 4️⃣ Google OCR 호출 (Vision API)
    with open("cropped.png", "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "requests": [{
            "image": {"content": img_b64},
            "features": [{"type": "TEXT_DETECTION"}]
        }]
    }
    res = requests.post(
        "https://vision.googleapis.com/v1/images:annotate?key=YOUR_GOOGLE_AI_API_KEY",
        json=payload
    )
    text = res.json()["responses"][0].get("fullTextAnnotation", {}).get("text", "")

    # 5️⃣ 단순 정규식으로 값 추출 예시
    import re
    card_no = re.search(r"\d{4}[-\s*]{0,1}\d{4}[-\s*]{0,1}\d{4}", text)
    amount = re.search(r"(\d{1,3}(?:,\d{3})*)\s*원", text)
    date = re.search(r"\d{4}-\d{2}-\d{2}", text)

    # 6️⃣ Base64로 이미지 반환
    with open("cropped.png", "rb") as img_f:
        img_base64 = base64.b64encode(img_f.read()).decode()

    return {
        "transaction_date": date.group(0) if date else None,
        "card_number": card_no.group(0) if card_no else None,
        "amount": amount.group(1) if amount else None,
        "cropped_image": f"data:image/png;base64,{img_base64}",
    }

@app.post("/ocr/upload")
async def blur_sensitive_info(file: UploadFile = File(...), user_id: str = "tester"):
    """
    PDF에서 민감 정보(카드번호, 8-10자리 숫자)를 자동으로 감지하고 블러 처리합니다.

    - PyMuPDF4LLM으로 PDF에서 이미지 추출
    - Tesseract OCR로 텍스트 위치 감지
    - 카드번호(15자리 이상), 8-10자리 숫자를 찾아 블러 처리
    - 블러 처리된 이미지를 Base64로 반환
    """
    # 작업 시작 시간 기록
    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d%H%M%S")
    job_uuid = str(uuid.uuid4())[:8]  # UUID의 앞 8자리만 사용

    # PDF 임시 저장 폴더 생성
    custom_temp_folder = f"static/temp/pdf/{user_id}_{timestamp}_{job_uuid}"
    os.makedirs(custom_temp_folder, exist_ok=True)

    # PDF 파일 임시 저장
    temp_pdf = tempfile.NamedTemporaryFile(dir=custom_temp_folder,delete=False, suffix=".pdf")
    temp_pdf.write(await file.read())
    temp_pdf.close()
    
    # OCR 결과 폴더명 생성
    output_folder = f"static/temp/ocr/{user_id}_{timestamp}_{job_uuid}"
    os.makedirs(output_folder, exist_ok=True)

    # 폴더 내 모든 PDF 파일 찾기
    pdf_files = glob.glob(os.path.join(custom_temp_folder, "*.pdf"))
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

        md_text = pymupdf4llm.to_markdown(PDF_PATH)
        md_chunks = pymupdf4llm.to_markdown(PDF_PATH, page_chunks=True)
        if isinstance(md_chunks, list) and md_chunks:
            print(f"   First chunk length: {len(md_chunks[0].get('text', ''))}")
            print(f"   First chunk preview: {md_chunks[0].get('text', '')[:300]}")

        pdf_name = os.path.splitext(os.path.basename(PDF_PATH))[0]
        temp_image_dir = f"static/temp/ocr/{pdf_name}"
        os.makedirs(temp_image_dir, exist_ok=True)

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

        if os.path.exists(temp_image_dir):
            image_files = [f for f in os.listdir(temp_image_dir) if f.endswith('.png')]
            for img_file in image_files:
                img_path = os.path.join(temp_image_dir, img_file)
                img = Image.open(img_path)

                ocr_text = pytesseract.image_to_string(img, lang='kor+eng', config='--psm 6')
                data = pytesseract.image_to_data(img, lang='kor+eng', config='--psm 6', output_type=pytesseract.Output.DICT)

                # 민감 정보 찾기: 카드번호 + 8~10자리 모든 숫자
                n_boxes = len(data['text'])
                positions = []

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

                    # 저장 - 파일명: 작업시작시간 + 원본파일명
                    output_filename = f"{timestamp}_{pdf_name}_{img_file}"
                    output_path = os.path.join(output_folder, output_filename)
                    blurred_img = Image.fromarray(blurred)
                    blurred_img.save(output_path)
        else:
            print("⚠️ No images extracted") 

    return {"message": f"블러 처리 완료! 결과는 {output_folder}에 저장되었습니다."}

@app.post("/merge/")
async def merge_pdfs_in_order(request: MergeRequest):
    """
    사용자가 지정한 순서대로 PDF 파일을 병합합니다.
    {
        "pdf_list": [
            "0601_1.pdf",
            "0601_2.pdf",
            "0602_1.pdf",
            "0605_1.pdf",
            "0605_2.pdf",
            "0607_1.pdf",
            "0607_2.pdf",
            "0608_1.pdf",
            "0608_2.pdf",
            "0609_1.pdf",
            "0609_2.pdf",
            "0612_1.pdf",
            "0612_2.pdf",
            "0612_3.pdf",
            "0613_1.pdf",
            "0613_2.pdf",
            "0613_3.pdf",
            "0614_1.pdf",
            "0614_2.pdf",
            "0615_1.pdf",
            "0615_2.pdf",
            "0616_1.pdf",
            "0616_2.pdf",
            "0616_3.pdf",
            "0619_1.pdf",
            "0619_2.pdf",
            "0619_3.pdf",
            "0620_1.pdf",
            "0620_2.pdf",
            "0621_1.pdf",
            "0621_2.pdf",
            "0622_1.pdf",
            "0622_2.pdf",
            "0623_1.pdf",
            "0623_2(시건장치).pdf",
            "0626_1.pdf",
            "0627_1.pdf",
            "0627_2.pdf",
            "0628_1.pdf",
            "0630_1.pdf",
            "0630_2(인터넷랜어댑터).pdf"
        ],
        "pdf_path": "tester_20251027160000_cf1500be",
        "output_file_name": "merged_result.pdf"
    }
    """
    merged = pikepdf.Pdf.new()
    total_pages = 0
    base_pdf_path = f"static/temp/{request.pdf_path}/pdf"
    result_folder = f"static/temp/{request.pdf_path}/result"
    os.makedirs(result_folder, exist_ok=True)

    for pdf_filename in request.pdf_list:
        full_path = os.path.join(base_pdf_path, pdf_filename)
        try:
            src = pikepdf.Pdf.open(full_path)
            merged.pages.extend(src.pages)
            total_pages += len(src.pages)
            src.close()
        except Exception as e:
            print(f"{full_path} 병합 실패: {e}")

    # 병합된 파일을 result 폴더에 저장
    result_path = os.path.join(result_folder, request.output_file_name)
    merged.save(result_path)
    merged.close()
    print(f"완료: {result_path} (총 {total_pages} 페이지)")

    # 파일 다운로드 응답 반환
    return FileResponse(
        path=result_path,
        media_type="application/pdf",
        filename=request.output_file_name
    )