import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os
import re


# --- 설정 (사용자 환경에 맞게 수정) ---

# 1. Tesseract 실행 파일 경로 (Windows 사용자만 해당)
#    Windows에 Tesseract를 설치한 경우, 아래 경로를 Tesseract.exe 파일이 있는 곳으로 수정해야 할 수 있습니다.
#    macOS나 Linux에서는 보통 자동으로 경로를 찾으므로 주석 처리해두어도 됩니다.
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 2. Poppler 경로 (Windows 사용자만 해당)
#    환경 변수 Path에 Poppler의 bin 폴더를 추가하지 않은 경우, 아래 경로를 직접 지정해야 합니다.
POPPLER_PATH = r"C:\Program Files\poppler-25.07.0\Library\bin"

def preprocess_for_numbers(image: Image.Image) -> Image.Image:
    """
    숫자 인식에 특화된 이미지 전처리 함수.
    카드번호, 승인번호 등 숫자 인식률을 높이기 위한 전처리를 수행합니다.

    :param image: PIL Image 객체
    :return: 전처리된 PIL Image 객체
    """
    # PIL 이미지를 OpenCV 배열로 변환
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

    # 1. 크기 확대 (숫자를 더 명확하게)
    scale_percent = 200  # 2배 확대
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)

    # 2. 대비 향상 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(resized)

    # 3. 이진화 (Otsu)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. 모폴로지 연산 (노이즈 제거 및 글자 선명화)
    kernel = np.ones((2,2), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 5. 샤프닝
    kernel_sharpen = np.array([[-1,-1,-1],
                                [-1, 9,-1],
                                [-1,-1,-1]])
    sharpened = cv2.filter2D(morph, -1, kernel_sharpen)

    return Image.fromarray(sharpened)

def fix_ocr_misreads(text: str) -> str:
    """
    OCR 오인식 문자를 숫자로 보정하는 함수.

    :param text: OCR로 추출한 원본 텍스트
    :return: 보정된 텍스트
    """
    # 문자 -> 숫자 매핑 (OCR 오인식 패턴)
    char_map = {
        # 0으로 오인식되는 문자들
        'O': '0', 'o': '0', 'Q': '0', 'D': '0',
        # 1로 오인식되는 문자들
        'I': '1', 'l': '1', '|': '1', 'i': '1', 'j': '1',
        # 2로 오인식되는 문자들
        'Z': '2', 'z': '2',
        # 3으로 오인식되는 문자들
        # (특별한 경우 없음)
        # 4로 오인식되는 문자들
        'A': '4',
        # 5로 오인식되는 문자들
        'S': '5', 's': '5',
        # 6으로 오인식되는 문자들
        'G': '6', 'g': '6', 'b': '6',
        # 7로 오인식되는 문자들
        'T': '7', 't': '7',
        # 8로 오인식되는 문자들
        'B': '8',
        # 9로 오인식되는 문자들
        'g': '9', 'q': '9',
        # 별표(*)로 오인식되는 문자들
        '×': '*', 'x': '*', 'X': '*',
    }

    corrected = text
    for old_char, new_char in char_map.items():
        corrected = corrected.replace(old_char, new_char)

    return corrected

def clean_card_number(raw_text: str) -> str:
    """
    OCR로 추출한 카드번호를 정제하는 함수.
    잘못 인식된 문자를 숫자로 보정합니다.

    :param raw_text: OCR로 추출한 원본 텍스트
    :return: 정제된 카드번호 (16자리)
    """
    # 1단계: 오인식 문자 보정
    corrected = fix_ocr_misreads(raw_text)

    # 2단계: 숫자, -, * 만 남기기
    cleaned = re.sub(r'[^0-9\-*]', '', corrected)

    # 3단계: 구분자 제거하고 16자리로 만들기
    # 예: 5236-12**-****-0051 → 523612**0051 (12자) → 5236-12**-****-0051 (16자)

    # 구분자 제거
    no_separator = cleaned.replace('-', '')

    # 16자리가 되도록 처리
    if len(no_separator) < 16:
        # 부족한 경우 중간을 *로 채우기
        missing = 16 - len(no_separator)
        # 앞 4자리 + * 채우기 + 뒤 4자리
        if len(no_separator) >= 8:
            # 앞뒤가 있으면 중간에 * 추가
            front = no_separator[:4]
            back = no_separator[-4:]
            middle_part = no_separator[4:-4]
            middle = (middle_part + '*' * missing)[:8]
            no_separator = front + middle + back
        else:
            # 전체가 짧으면 뒤에 * 추가
            no_separator = (no_separator + '*' * missing)[:16]
    elif len(no_separator) > 16:
        # 초과하는 경우 16자리만 사용
        no_separator = no_separator[:16]

    # 4단계: 4자리씩 끊어서 구분자 추가
    result = '-'.join([no_separator[i:i+4] for i in range(0, 16, 4)])

    return result

def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """
    OCR 정확도를 높이기 위해 이미지를 전처리하는 함수.
    기울기 보정, 회색조 변환, 이진화를 포함합니다.

    :param image: PIL Image 객체
    :return: 전처리된 PIL Image 객체
    """
    # PIL 이미지를 OpenCV에서 처리할 수 있는 numpy 배열로 변환
    open_cv_image = np.array(image)
    print(f"원본 이미지 크기: {open_cv_image.shape[1]} x {open_cv_image.shape[0]} (폭 x 높이)")

    # 1. 회색조 변환
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    # 2. 기울기 보정
    #    텍스트가 아닌 영역을 검게, 텍스트를 희게 반전시켜 윤곽선을 더 잘 찾도록 함
    inverted = cv2.bitwise_not(gray)
    #    Otsu의 이진화를 통해 텍스트 영역을 명확히 함
    # thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # 작은 노이즈 제거
    thresh = cv2.medianBlur(thresh, 3)

    #    텍스트 영역의 좌표를 찾음
    coords = np.column_stack(np.where(thresh > 0))
    #    최소한의 면적을 가지는 회전된 사각형으로 텍스트 영역을 감쌈
    angle = cv2.minAreaRect(coords)[-1]

    #    cv2.minAreaRect는 -90도에서 0도 사이의 값을 반환함
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    #    계산된 각도만큼 이미지 전체를 회전시킴
    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    print(f"이미지 기울기 보정 완료. 감지된 각도: {angle:.2f}도")

    # 3. 이진화 (Otsu의 알고리즘 사용)
    #    회전된 이미지에 다시 한번 Otsu의 이진화를 적용하여 최종적으로 텍스트를 선명하게 함
    final_thresh = cv2.threshold(rotated, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # (선택 사항) 4. 노이즈 제거
    #    만약 이미지에 작은 점 같은 노이즈가 많다면 아래 코드의 주석을 해제하여 사용
    # final_thresh = cv2.medianBlur(final_thresh, 3)

    # OpenCV 이미지를 다시 PIL 이미지로 변환하여 반환
    return Image.fromarray(final_thresh)


def mask_sensitive_areas(image_np: np.ndarray, label_positions: list) -> np.ndarray:
    """
    이미지에서 라벨 텍스트만 마스킹하는 함수.

    :param image_np: OpenCV numpy 배열 (이미지)
    :param label_positions: 마스킹할 라벨들의 위치 정보 리스트 [(label, x, y, w, h), ...]
    :return: 마스킹된 이미지
    """
    masked = image_np.copy()

    for label, x, y, w, h in label_positions:
        # 라벨 텍스트만 마스킹 (약간의 여유만 추가)
        expand = 5  # 모든 방향으로 5픽셀만 확장

        # 확장된 박스 계산
        x1 = max(0, x - expand)
        y1 = max(0, y - expand)
        x2 = min(masked.shape[1], x + w + expand)
        y2 = min(masked.shape[0], y + h + expand)

        # 흰색 사각형으로 마스킹
        cv2.rectangle(masked, (x1, y1), (x2, y2), (255, 255, 255), -1)

        print(f"  [마스킹] {label} 라벨 텍스트 마스킹 완료: ({x1}, {y1}) → ({x2}, {y2})")

    return masked

def extract_text_from_pdf(pdf_path: str, lang: str = 'kor+eng') -> str:
    """
    PDF 파일에서 텍스트를 추출하는 메인 함수.
    내부적으로 PDF를 이미지로 변환하고, 각 이미지를 전처리한 후 OCR을 수행합니다.

    :param pdf_path: 텍스트를 추출할 PDF 파일의 경로
    :param lang: Tesseract가 사용할 언어 ('kor', 'eng', 'kor+eng' 등)
    :return: PDF의 모든 페이지에서 추출된 전체 텍스트
    """
    if not os.path.exists(pdf_path):
        return f"오류: 파일을 찾을 수 없습니다 - {pdf_path}"

    print(f"'{pdf_path}' 파일 처리 시작...")

    # --- 단계 1: Poppler를 사용하여 PDF를 고해상도 이미지로 변환 ---
    try:
        # Windows에서 poppler_path를 직접 지정
        images = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)
    except Exception as e:
        return f"PDF를 이미지로 변환하는 중 오류 발생: {e}\nPoppler가 시스템에 설치되어 있고 Path에 등록되었는지 확인하세요."

    full_text = ""

    # --- 단계 2 & 3: 각 페이지 이미지를 전처리하고 Tesseract로 텍스트 추출 ---
    for i, image in enumerate(images):
        page_num = i + 1
        print(f"\n--- {page_num}페이지 처리 중 ---")

        # 단계 2: 이미지 전처리
        preprocessed_image = preprocess_image_for_ocr(image)

        # (디버깅용) 전처리된 이미지를 파일로 저장
        preprocessed_image.save(f'preprocessed_general_page_{page_num}.png')
        print(f"  일반 전처리 이미지 저장: preprocessed_general_page_{page_num}.png")

        # 단계 3: Tesseract OCR 실행
        # --oem 3: 기본 LSTM 엔진 사용
        # --psm 3: 자동 페이지 분할 (대부분의 경우 가장 안정적)
        custom_config = r'--oem 3 --psm 11'
        # -- psm 3: (기본값) 완전 자동 페이지 분할. 대부분의 문서에 잘 작동합니다.
        # -- psm 4 : 문서가 단일 텍스트 열(column)이라고 가정.
        # -- psm 6: 문서가 균일한 단일 텍스트 블록이라고 가정 (예: 소설책 한 페이지).
        # -- psm 11 : 텍스트가 드문드문 흩어져 있는 경우 (예: 영수증).
        # -- psm 12: OSD(방향 및 스크립트 감지)만 포함된 sparse text.

        try:
            # 텍스트 추출 (기존)
            text = pytesseract.image_to_string(preprocessed_image, lang=lang, config=custom_config)

            # 위치 정보와 함께 데이터 추출
            data = pytesseract.image_to_data(preprocessed_image, lang=lang, config=custom_config, output_type=pytesseract.Output.DICT)

            # 숫자 특화 전처리 이미지로 추가 OCR 수행 (카드번호, 승인번호용)
            print(f"  숫자 인식 특화 OCR 실행 중...")
            preprocessed_numbers = preprocess_for_numbers(image)

            # (디버깅용) 숫자 특화 전처리 이미지 저장
            preprocessed_numbers.save(f'preprocessed_numbers_page_{page_num}.png')
            print(f"  숫자 특화 전처리 이미지 저장: preprocessed_numbers_page_{page_num}.png")

            # 숫자 전용 설정: PSM 6 (균일한 텍스트 블록) + 숫자/기호 화이트리스트
            number_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789-*'
            data_numbers = pytesseract.image_to_data(preprocessed_numbers, lang='eng', config=number_config, output_type=pytesseract.Output.DICT)

            # 페이지별 결과 저장
            full_text += f"\n\n--- 페이지 {page_num} ---\n"
            full_text += text
            full_text += "\n\n[추출된 구조화 데이터]\n"

            # OCR 결과에서 특정 패턴 찾기
            n_boxes = len(data['text'])

            # 추출할 데이터 초기화
            transaction_date = None
            transaction_date_bbox = None
            card_number = None
            card_number_bbox = None
            card_number_raw = None
            approval_number = None
            approval_number_bbox = None
            store_name = None
            store_name_bbox = None

            # 1차: 일반 OCR 결과에서 찾기
            for i in range(n_boxes):
                if int(data['conf'][i]) < 20:  # 신뢰도가 낮은 텍스트는 건너뛰기
                    continue

                txt = data['text'][i].strip()
                if not txt:
                    continue

                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

                # 거래일자 (YYYY-MM-DD HH:MM:SS)
                if not transaction_date:
                    date_match = re.search(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}', txt)
                    if date_match:
                        transaction_date = date_match.group(0)
                        transaction_date_bbox = (x, y, w, h)
                        print(f"  [거래일자] {transaction_date} (위치: {transaction_date_bbox})")

                # 카드 번호 패턴 찾기 (매우 유연한 패턴)
                if not card_number:
                    # 최소 15자 이상, 숫자/*/×/문자 혼합 패턴
                    if len(txt) >= 15:
                        card_match = re.search(r'[\d*OoIlxX×]{2,5}[-\s][\d*OoIlxX×]{2,5}[-\s][\d*OoIlxX×]{2,5}[-\s][\d*OoIlxX×]{2,5}', txt)
                        if card_match:
                            card_number_raw = card_match.group(0)
                            card_number = clean_card_number(card_number_raw)
                            card_number_bbox = (x, y, w, h)
                            print(f"  [카드번호] 원본: {card_number_raw} → 보정: {card_number}")

                # 승인번호 (8자리 숫자)
                if not approval_number:
                    approval_match = re.match(r"^\d{8}$", txt)
                    if approval_match:
                        approval_number = txt
                        approval_number_bbox = (x, y, w, h)
                        print(f"  [승인번호] {approval_number} (위치: {approval_number_bbox})")

                # 가맹점명 찾기 ("가맹점명" 다음 줄)
                if not store_name and "가맹점" in txt:
                    # 다음 box 확인
                    if i + 1 < n_boxes:
                        next_txt = data['text'][i + 1].strip()
                        if next_txt and int(data['conf'][i + 1]) >= 20:
                            store_name = next_txt
                            store_name_bbox = (data['left'][i + 1], data['top'][i + 1],
                                             data['width'][i + 1], data['height'][i + 1])
                            print(f"  [가맹점명] {store_name} (위치: {store_name_bbox})")

            # 2차: 숫자 특화 OCR 결과에서 카드번호 찾기 (일반 OCR에서 못 찾은 경우)
            if not card_number:
                n_boxes_num = len(data_numbers['text'])
                for i in range(n_boxes_num):
                    if int(data_numbers['conf'][i]) < 40:
                        continue

                    txt = data_numbers['text'][i].strip()
                    if not txt:
                        continue

                    x, y, w, h = data_numbers['left'][i], data_numbers['top'][i], data_numbers['width'][i], data_numbers['height'][i]

                    # 카드번호 패턴 (최소 15자 이상)
                    if len(txt) >= 15:
                        card_match = re.search(r'[\d*-]{15,}', txt)
                        if card_match:
                            card_number_raw = card_match.group(0)
                            card_number = clean_card_number(card_number_raw)
                            card_number_bbox = (x, y, w, h)
                            print(f"  [숫자 특화 OCR] 원본: {card_number_raw} → 보정: {card_number}")
                            break

            # 3차: 전체 텍스트에서 직접 검색 (box 단위로는 분리되었을 경우 대비)
            # 카드번호
            if not card_number:
                print(f"  [전체 텍스트 검색] 카드번호 패턴 찾는 중...")
                full_text_search = text.replace('\n', ' ')
                card_match = re.search(r'[\d*OoIlxX×]{3,5}[-\s]*[\d*OoIlxX×]{3,5}[-\s]*[\d*OoIlxX×]{3,5}[-\s]*[\d*OoIlxX×]{3,5}', full_text_search)
                if card_match:
                    card_number_raw = card_match.group(0)
                    card_number = clean_card_number(card_number_raw)
                    card_number_bbox = (0, 0, 0, 0)
                    print(f"  [전체 텍스트 검색] 원본: {card_number_raw} → 보정: {card_number}")

            # 거래일자 (전체 텍스트에서 검색)
            if not transaction_date:
                print(f"  [전체 텍스트 검색] 거래일자 패턴 찾는 중...")
                date_match = re.search(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}', text)
                if date_match:
                    transaction_date = date_match.group(0)
                    transaction_date_bbox = (0, 0, 0, 0)
                    print(f"  [전체 텍스트 검색] {transaction_date}")

            # 가맹점명 (전체 텍스트에서 "가맹점명" 다음 줄 찾기)
            if not store_name:
                print(f"  [전체 텍스트 검색] 가맹점명 찾는 중...")
                lines = text.split('\n')
                for idx, line in enumerate(lines):
                    if '가맹점' in line and '명' in line:
                        # 다음 줄 확인
                        if idx + 1 < len(lines):
                            next_line = lines[idx + 1].strip()
                            if next_line and len(next_line) > 0:
                                store_name = next_line
                                store_name_bbox = (0, 0, 0, 0)
                                print(f"  [전체 텍스트 검색] {store_name}")
                                break

            # 민감 정보 라벨 위치 찾기 (마스킹용)
            print(f"\n  민감 정보 라벨 위치 검색 중...")
            label_positions = []

            # 전체 텍스트에서도 라벨 검색 (디버깅용)
            print(f"  [디버그] OCR 전체 텍스트 일부:")
            for line in text.split('\n')[:20]:  # 처음 20줄만
                if line.strip():
                    print(f"    {line}")

            for i in range(n_boxes):
                if int(data['conf'][i]) < 10:  # 신뢰도 낮춰서 더 많이 검색
                    continue

                txt = data['text'][i].strip()
                if not txt:
                    continue

                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

                # "카드번호" 또는 "카드 번호" 라벨 찾기 (더 유연한 검색)
                if '카드' in txt or '번호' in txt:
                    # 주변 박스들도 확인하여 "카드번호"를 구성하는지 확인
                    combined_text = txt
                    # 다음 박스 확인
                    if i + 1 < n_boxes:
                        next_txt = data['text'][i + 1].strip()
                        combined_text += next_txt

                    if '카드' in combined_text and '번호' in combined_text:
                        label_positions.append(('카드번호', x, y, w, h))
                        print(f"  [라벨 발견] 카드번호 위치: ({x}, {y}, {w}, {h}) - 텍스트: '{txt}'")

                # "승인번호" 또는 "승인 번호" 라벨 찾기
                if '승인' in txt or ('인' in txt and i > 0):
                    combined_text = txt
                    # 다음 박스 확인
                    if i + 1 < n_boxes:
                        next_txt = data['text'][i + 1].strip()
                        combined_text += next_txt

                    if '승인' in combined_text and '번호' in combined_text:
                        label_positions.append(('승인번호', x, y, w, h))
                        print(f"  [라벨 발견] 승인번호 위치: ({x}, {y}, {w}, {h}) - 텍스트: '{txt}'")

            # 원본 이미지를 numpy 배열로 변환하여 마스킹
            if label_positions:
                print(f"\n  민감 정보 영역 마스킹 시작... (총 {len(label_positions)}개 라벨)")
                original_image_np = np.array(image)
                masked_image = mask_sensitive_areas(original_image_np, label_positions)

                # 마스킹된 이미지 저장
                masked_pil = Image.fromarray(masked_image)
                masked_filename = f'masked_page_{page_num}.png'
                masked_pil.save(masked_filename)
                print(f"  마스킹된 이미지 저장: {masked_filename}")
            else:
                print(f"  민감 정보 라벨을 찾지 못했습니다.")

            # 추출된 데이터 출력
            if transaction_date:
                full_text += f"거래일자: {transaction_date}\n"
                full_text += f"  위치: x={transaction_date_bbox[0]}, y={transaction_date_bbox[1]}, w={transaction_date_bbox[2]}, h={transaction_date_bbox[3]}\n"

            if card_number:
                full_text += f"카드번호: {card_number}\n"
                full_text += f"  위치: x={card_number_bbox[0]}, y={card_number_bbox[1]}, w={card_number_bbox[2]}, h={card_number_bbox[3]}\n"

            if approval_number:
                full_text += f"승인번호: {approval_number}\n"
                full_text += f"  위치: x={approval_number_bbox[0]}, y={approval_number_bbox[1]}, w={approval_number_bbox[2]}, h={approval_number_bbox[3]}\n"

            if store_name:
                full_text += f"가맹점명: {store_name}\n"
                full_text += f"  위치: x={store_name_bbox[0]}, y={store_name_bbox[1]}, w={store_name_bbox[2]}, h={store_name_bbox[3]}\n"

            print(f"\n{page_num}페이지 OCR 완료.")
            print(f"=== 추출 결과 ===")
            if transaction_date:
                print(f"  거래일자: {transaction_date}")
            if card_number:
                print(f"  카드번호: {card_number}")
            if approval_number:
                print(f"  승인번호: {approval_number}")
            if store_name:
                print(f"  가맹점명: {store_name}")

        except pytesseract.TesseractNotFoundError:
            return "Tesseract 오류: Tesseract가 설치되지 않았거나 시스템 Path에 등록되지 않았습니다."
        except Exception as e:
            return f"{page_num}페이지 OCR 중 오류 발생: {e}"

    print("\n모든 페이지 처리 완료.")
    return full_text


if __name__ == '__main__':
    # 여기에 OCR을 수행할 PDF 파일 경로를 입력하세요.
    pdf_file_path = "C:/Users/user/Desktop/expense_cliaim/clip.pdf"
    # PDF에서 텍스트 추출 실행
    extracted_text = extract_text_from_pdf(pdf_file_path, lang='kor+eng')

    # 결과 출력
    # print("\n\n========== OCR 최종 결과 ==========")
    # print(extracted_text)

    # 결과를 텍스트 파일로 저장
    # output_file_path = f"{os.path.splitext(pdf_file_path)[0]}_result.txt"
    # with open(output_file_path, 'w', encoding='utf-8') as f:
    #     f.write(extracted_text)
    # print(f"\n결과가 '{output_file_path}' 파일에 저장되었습니다.")