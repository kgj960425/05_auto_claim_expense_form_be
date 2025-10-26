from fastapi import FastAPI, File, UploadFile
import tempfile, os
import cv2
import numpy as np
import fitz  # PyMuPDF
import base64
import requests

from fastapi.responses import JSONResponse
from utils.crop_receipt import auto_crop_receipt

app = FastAPI()

@app.get("/serverCheck")
def custom_json():
    data = {"msg": "ok"}
    return JSONResponse(content=data, status_code=200)

@app.post("/ocr/receipt")
async def process_receipt(file: UploadFile = File(...)):
    # 1️⃣ PDF 임시 저장
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
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
