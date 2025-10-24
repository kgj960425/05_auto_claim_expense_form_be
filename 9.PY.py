import re
import cv2
import numpy as np
from pdf2image import convert_from_path
import pytesseract
import easyocr

# ==========================
# 설정
# ==========================
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\Program Files\poppler-25.07.0\Library\bin"
PDF_PATH = r"C:\Users\user\Desktop\expense_cliaim\3.pdf"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ==========================
# 1️⃣ PDF → 이미지 변환
# ==========================
pages = convert_from_path(PDF_PATH, dpi=300, poppler_path=POPPLER_PATH)


# ==========================
# 2️⃣ 이미지 전처리 + OCR
# ==========================
def preprocess_image(page):
    """OCR 인식률을 높이기 위한 고급 전처리"""
    img = np.array(page)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 대비 향상
    gray = cv2.equalizeHist(gray)

    # 적응형 이진화 (회색 배경 영수증에 효과적)
    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )

    # 노이즈 제거 및 글자 강조
    kernel = np.ones((1, 1), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    gray = cv2.medianBlur(gray, 3)

    # 확대 (작은 글씨 강화)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    return gray


def ocr_extract_text(page):
    """Tesseract + EasyOCR 병행"""
    gray = preprocess_image(page)

    # Tesseract OCR
    tesseract_text = pytesseract.image_to_string(gray, lang="kor+eng", config="--psm 6")

    # EasyOCR OCR
    reader = easyocr.Reader(["ko", "en"], gpu=False)
    easy_text = "\n".join(reader.readtext(gray, detail=0))

    # 두 결과를 합치고 중복 제거
    combined_text = "\n".join(
        set(tesseract_text.splitlines() + easy_text.splitlines())
    )
    return combined_text


# 전체 페이지 OCR 수행
full_text = ""
for i, page in enumerate(pages):
    page_text = ocr_extract_text(page)
    full_text += page_text + "\n"
    print(f"\n[Page {i+1}] OCR 추출 완료, 길이: {len(page_text)}")


# ==========================
# 3️⃣ 필요한 정보 정규식 추출 (유연 매칭)
# ==========================
patterns = {
    "거래일자": [
        re.compile(
            r"(거래.?일.?자|일시)\s*[:\-]?\s*([\d]{4}[-./]?\d{2}[-./]?\d{2}[\sT]?\d{2}[:]\d{2}[:]\d{2})"
        ),
        re.compile(r"([\d]{4}[-./]\d{2}[-./]\d{2}\s*\d{2}[:]\d{2}[:]\d{2})"),
    ],
    "카드번호": [
        re.compile(
            r"(카드.?번호|카드.?명)\s*[:\-]?\s*([0-9*]{4,}[- ]?[0-9*]{2,}[- ]?[0-9*]{2,}[- ]?[0-9*]{4,})"
        ),
        re.compile(r"([0-9*]{4}[- ]?[0-9*]{2,}[- ]?[0-9*]{2,}[- ]?[0-9*]{4})"),
    ],
    "승인번호": [
        re.compile(r"(승인.?번호)\s*[:\-]?\s*([0-9]{6,10})"),
        re.compile(r"\b([0-9]{6,10})\b"),
    ],
    "가맹점명": [
        re.compile(r"(가맹.?점.?명)\s*[:\-]?\s*([가-힣A-Za-z0-9·\s]+)")
    ],
}

result = {}
for key, plist in patterns.items():
    for pattern in plist:
        match = pattern.search(full_text)
        if match:
            result[key] = match.groups()[-1].strip()
            break


# ==========================
# 4️⃣ 결과 출력
# ==========================
print("\n=== OCR 추출 결과 ===")
for k, v in result.items():
    print(f"{k}: {v}")

# ==========================
# 5️⃣ 디버그용 전체 텍스트 출력
# ==========================
print("\n=== RAW TEXT (일부 미리보기) ===")
print(full_text[:2000])
