import re
import cv2
import numpy as np
from pdf2image import convert_from_path
import pytesseract
import easyocr
from PIL import Image
from crop_receipt import auto_crop_receipt

# ==========================
# 설정
# ==========================
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\Program Files\poppler-25.07.0\Library\bin"
PDF_PATH = r"C:\Users\user\Desktop\expense_cliaim\1.pdf"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ==========================
# 1️⃣ PDF → 이미지 변환 + Crop
# ==========================
pages = convert_from_path(PDF_PATH, dpi=300, poppler_path=POPPLER_PATH)

# Crop 처리된 페이지 저장
cropped_pages = []
for i, page in enumerate(pages):
    # 임시 파일로 저장
    temp_path = f"temp_page_{i+1}.png"
    page.save(temp_path, "PNG")

    # Crop 적용
    cropped_path = f"cropped_page_{i+1}.png"
    try:
        cropped_img = auto_crop_receipt(temp_path, cropped_path, show=False)
        # OpenCV BGR → PIL RGB 변환
        cropped_pil = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
        cropped_pages.append(cropped_pil)
        print(f"✅ Page {i+1} Crop 완료: {cropped_img.shape[1]} x {cropped_img.shape[0]}")
    except Exception as e:
        print(f"⚠️ Page {i+1} Crop 실패, 원본 사용: {e}")
        cropped_pages.append(page)

# 원본 pages를 cropped_pages로 교체
pages = cropped_pages


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


def ocr_extract_with_position(page):
    """Tesseract image_to_data로 위치 정보 포함 추출"""
    gray = preprocess_image(page)

    # Tesseract로 위치 정보와 함께 텍스트 추출
    data = pytesseract.image_to_data(gray, lang="kor+eng", config="--psm 6", output_type=pytesseract.Output.DICT)

    # EasyOCR로 위치 정보와 함께 텍스트 추출
    reader = easyocr.Reader(["ko", "en"], gpu=False)
    easy_results = reader.readtext(gray, detail=1)

    return data, easy_results, gray


def find_sensitive_info_positions(data, easy_results, card_number, approval_number):
    """카드번호와 승인번호의 위치 찾기 (정확한 매칭만)"""
    positions = []

    # 카드번호와 승인번호를 숫자만 추출
    card_digits = re.sub(r'[^0-9*]', '', card_number) if card_number else ""
    approval_digits = re.sub(r'[^0-9]', '', approval_number) if approval_number else ""

    print(f"  검색 대상 - 카드번호: {card_digits}, 승인번호: {approval_digits}")
    print(f"  승인번호 길이: {len(approval_digits)}")

    # Tesseract 데이터에서 검색
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        if int(data['conf'][i]) < 30:  # 신뢰도가 낮은 텍스트는 건너뛰기
            continue

        text = data['text'][i].strip()
        if not text:
            continue

        text_digits = re.sub(r'[^0-9*]', '', text)

        # 카드번호 정확한 매칭 (최소 12자리 이상 & 완전 일치)
        if card_digits and len(text_digits) >= 12:
            # 카드번호 패턴 확인 (16자리 또는 15자리)
            if len(text_digits) >= 15 and text_digits == card_digits:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                positions.append(('카드번호', x, y, w, h))
                print(f"  🔍 카드번호 위치 발견: ({x}, {y}, {w}, {h}) - 텍스트: {text}")

        # 승인번호 정확한 매칭 (8자리만 - 일반적인 승인번호 형식)
        if approval_digits and len(approval_digits) == 8:
            # 숫자만 추출한 텍스트가 정확히 8자리이고 승인번호와 일치하는 경우만
            if len(text_digits) == 8 and text_digits == approval_digits:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                positions.append(('승인번호', x, y, w, h))
                print(f"  🔍 승인번호 위치 발견: ({x}, {y}, {w}, {h}) - 텍스트: {text}")
                print(f"      매칭: text_digits={text_digits}, approval_digits={approval_digits}")

    # EasyOCR 결과에서 검색
    for bbox, text, conf in easy_results:
        if conf < 0.3:
            continue

        text_digits = re.sub(r'[^0-9*]', '', text)

        # 카드번호 정확한 매칭
        if card_digits and len(text_digits) >= 12:
            if len(text_digits) >= 15 and text_digits == card_digits:
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                x, y = int(min(x_coords)), int(min(y_coords))
                w, h = int(max(x_coords) - x), int(max(y_coords) - y)
                positions.append(('카드번호', x, y, w, h))
                print(f"  🔍 [EasyOCR] 카드번호 위치 발견: ({x}, {y}, {w}, {h}) - 텍스트: {text}")

        # 승인번호 정확한 매칭 (8자리만)
        if approval_digits and len(approval_digits) == 8:
            if len(text_digits) == 8 and text_digits == approval_digits:
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                x, y = int(min(x_coords)), int(min(y_coords))
                w, h = int(max(x_coords) - x), int(max(y_coords) - y)
                positions.append(('승인번호', x, y, w, h))
                print(f"  🔍 [EasyOCR] 승인번호 위치 발견: ({x}, {y}, {w}, {h}) - 텍스트: {text}")
                print(f"      매칭: text_digits={text_digits}, approval_digits={approval_digits}")

    # 중복 제거 (같은 위치의 중복 검출 제거)
    unique_positions = []
    for pos in positions:
        label, x, y, w, h = pos
        # 이미 추가된 위치와 너무 가까우면 스킵
        is_duplicate = False
        for existing in unique_positions:
            ex_label, ex_x, ex_y, _, _ = existing
            if label == ex_label and abs(x - ex_x) < 20 and abs(y - ex_y) < 20:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_positions.append(pos)

    return unique_positions


def apply_blur_to_positions(image, positions, scale_factor=1.5):
    """지정된 위치에 블러 처리 적용"""
    img_array = np.array(image)
    blurred = img_array.copy()

    for label, x, y, w, h in positions:
        # 전처리 시 1.5배 확대했으므로 좌표도 스케일 조정
        x = int(x / scale_factor)
        y = int(y / scale_factor)
        w = int(w / scale_factor)
        h = int(h / scale_factor)

        # 여유 공간 추가
        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_array.shape[1], x + w + padding)
        y2 = min(img_array.shape[0], y + h + padding)

        # 해당 영역만 블러 처리
        roi = blurred[y1:y2, x1:x2]
        if roi.size > 0:
            blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
            blurred[y1:y2, x1:x2] = blurred_roi
            print(f"  ✅ {label} 블러 처리 완료: ({x1}, {y1}) → ({x2}, {y2})")

    return Image.fromarray(blurred)


# 전체 페이지 OCR 수행
full_text = ""
page_data_list = []  # 각 페이지의 OCR 데이터 저장

for i, page in enumerate(pages):
    print(f"\n[Page {i+1}] OCR 추출 시작...")
    page_text = ocr_extract_text(page)
    full_text += page_text + "\n"

    # 위치 정보 포함 OCR 수행
    data, easy_results, gray = ocr_extract_with_position(page)
    page_data_list.append((page, data, easy_results))

    print(f"[Page {i+1}] OCR 추출 완료, 길이: {len(page_text)}")


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
# 5️⃣ 민감 정보 위치 찾기 및 블러 처리
# ==========================
card_number = result.get("카드번호", "")
approval_number = result.get("승인번호", "")

if card_number or approval_number:
    print("\n=== 민감 정보 위치 검색 및 블러 처리 ===")

    for i, (page, data, easy_results) in enumerate(page_data_list):
        print(f"\n[Page {i+1}] 민감 정보 검색 중...")

        # 민감 정보 위치 찾기
        positions = find_sensitive_info_positions(data, easy_results, card_number, approval_number)

        if positions:
            print(f"[Page {i+1}] {len(positions)}개 위치 발견, 블러 처리 시작...")

            # 블러 처리 적용
            blurred_page = apply_blur_to_positions(page, positions)

            # 블러 처리된 이미지 저장
            output_path = f"blurred_page_{i+1}.png"
            blurred_page.save(output_path)
            print(f"✅ 블러 처리된 이미지 저장: {output_path}")
        else:
            print(f"[Page {i+1}] 민감 정보 위치를 찾지 못했습니다.")
else:
    print("\n⚠️ 카드번호 또는 승인번호를 추출하지 못했습니다.")


# ==========================
# 6️⃣ 디버그용 전체 텍스트 출력
# ==========================
print("\n=== RAW TEXT (일부 미리보기) ===")
print(full_text[:2000])
