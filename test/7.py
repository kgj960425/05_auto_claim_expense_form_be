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

        # (디버깅용) 전처리된 이미지를 파일로 저장해서 확인하고 싶을 경우 아래 주석 해제
        # preprocessed_image.save(f'preprocessed_page_{page_num}.png')

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

            # 페이지별 결과 저장
            full_text += f"\n\n--- 페이지 {page_num} ---\n"
            full_text += text
            full_text += "\n\n[추출된 구조화 데이터]\n"

            # OCR 결과에서 특정 패턴 찾기
            n_boxes = len(data['text'])

            card_number = None
            card_number_bbox = None
            approval_number = None
            approval_number_bbox = None

            for i in range(n_boxes):
                if int(data['conf'][i]) < 30:  # 신뢰도가 낮은 텍스트는 건너뛰기
                    continue

                txt = data['text'][i].strip()
                if not txt:
                    continue

                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

                # 카드 번호 (1234-5678-9012-3456 또는 1234-****-****-3456)
                if not card_number:
                    import re
                    card_match = re.search(r"\d{4}[-\s*]+\d{4}[-\s*]+\d{4}[-\s*]+\d{4}", txt)
                    if card_match:
                        card_number = card_match.group(0)
                        card_number_bbox = (x, y, w, h)

                # 승인번호 (보통 8자리 숫자)
                if not approval_number:
                    import re
                    if re.match(r"^\d{8}$", txt):
                        approval_number = txt
                        approval_number_bbox = (x, y, w, h)

            # 추출된 데이터 출력
            if card_number:
                full_text += f"카드번호: {card_number}\n"
                full_text += f"  위치: x={card_number_bbox[0]}, y={card_number_bbox[1]}, w={card_number_bbox[2]}, h={card_number_bbox[3]}\n"

            if approval_number:
                full_text += f"승인번호: {approval_number}\n"
                full_text += f"  위치: x={approval_number_bbox[0]}, y={approval_number_bbox[1]}, w={approval_number_bbox[2]}, h={approval_number_bbox[3]}\n"

            print(f"{page_num}페이지 OCR 완료.")
            if card_number:
                print(f"  카드번호: {card_number} (위치: {card_number_bbox})")
            if approval_number:
                print(f"  승인번호: {approval_number} (위치: {approval_number_bbox})")

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
    print("\n\n========== OCR 최종 결과 ==========")
    print(extracted_text)

    # 결과를 텍스트 파일로 저장
    # output_file_path = f"{os.path.splitext(pdf_file_path)[0]}_result.txt"
    # with open(output_file_path, 'w', encoding='utf-8') as f:
    #     f.write(extracted_text)
    # print(f"\n결과가 '{output_file_path}' 파일에 저장되었습니다.")