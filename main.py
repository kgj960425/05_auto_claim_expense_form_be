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
async def process_receipt(file: UploadFile = File(...)):
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

@app.post("/ocr/blur-sensitive-info")
async def blur_sensitive_info(file: UploadFile = File(...)):
    """
    PDF에서 민감 정보(카드번호, 8-10자리 숫자)를 자동으로 감지하고 블러 처리합니다.

    - PyMuPDF4LLM으로 PDF에서 이미지 추출
    - Tesseract OCR로 텍스트 위치 감지
    - 카드번호(15자리 이상), 8-10자리 숫자를 찾아 블러 처리
    - 블러 처리된 이미지를 Base64로 반환
    """
    # 1️⃣ PDF 임시 저장
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_pdf.write(await file.read())
    temp_pdf.close()

    # 2️⃣ 임시 이미지 디렉토리 생성
    temp_image_dir = tempfile.mkdtemp()

    try:
        # 3️⃣ PyMuPDF4LLM으로 이미지 추출
        pymupdf4llm.to_markdown(
            temp_pdf.name,
            write_images=True,
            image_path=temp_image_dir,
            page_chunks=True
        )

        # 4️⃣ 추출된 이미지 파일 찾기
        image_files = [f for f in os.listdir(temp_image_dir) if f.endswith('.png')]

        if not image_files:
            return {"error": "No images extracted from PDF"}

        blurred_images = []

        # 5️⃣ 각 이미지에 대해 OCR + 블러 처리
        for img_file in image_files:
            img_path = os.path.join(temp_image_dir, img_file)
            img = Image.open(img_path)

            # OCR로 텍스트 위치 정보 추출
            data = pytesseract.image_to_data(
                img,
                lang='kor+eng',
                config='--psm 6',
                output_type=pytesseract.Output.DICT
            )

            # 민감 정보 위치 찾기
            n_boxes = len(data['text'])
            positions = []
            detected_items = []

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
                    detected_items.append({'type': '카드번호', 'text': text, 'position': {'x': x, 'y': y}})

                # 8~10자리 숫자 (날짜 제외)
                elif 8 <= len(text_digits) <= 10:
                    if not re.match(r'20\d{2}[-/]\d{2}[-/]\d{2}', text):
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        positions.append((f'{len(text_digits)}자리', x, y, w, h))
                        detected_items.append({'type': f'{len(text_digits)}자리 숫자', 'text': text, 'position': {'x': x, 'y': y}})

            # 6️⃣ 블러 처리
            img_array = np.array(img)
            blurred = img_array.copy()

            for _, x, y, w, h in positions:
                padding = 10
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(blurred.shape[1], x + w + padding)
                y2 = min(blurred.shape[0], y + h + padding)

                roi = blurred[y1:y2, x1:x2]
                if roi.size > 0:
                    blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
                    blurred[y1:y2, x1:x2] = blurred_roi

            # 7️⃣ 블러 처리된 이미지를 Base64로 변환
            blurred_img = Image.fromarray(blurred)

            # 메모리에서 Base64 변환
            import io
            buffer = io.BytesIO()
            blurred_img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()

            blurred_images.append({
                'filename': img_file,
                'detected_count': len(positions),
                'detected_items': detected_items,
                'blurred_image': f"data:image/png;base64,{img_base64}"
            })

        return {
            "success": True,
            "total_images": len(image_files),
            "images": blurred_images
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        # 8️⃣ 임시 파일 정리
        os.unlink(temp_pdf.name)
        import shutil
        if os.path.exists(temp_image_dir):
            shutil.rmtree(temp_image_dir)

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