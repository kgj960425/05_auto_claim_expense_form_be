"""
OCR 처리 및 민감 정보 블러 처리 유틸리티
"""
import os
import re
import cv2
import numpy as np
import pytesseract
from PIL import Image
from typing import List, Tuple, Dict, Any, Optional


class OCRConfig:
    """OCR 설정 상수"""
    MIN_CONFIDENCE = 30
    BLUR_PADDING = 10
    BLUR_KERNEL_SIZE = (51, 51)
    CARD_NUMBER_MIN_LENGTH = 15
    SENSITIVE_NUMBER_MIN_LENGTH = 8
    SENSITIVE_NUMBER_MAX_LENGTH = 10
    TESSERACT_LANG = 'kor+eng'
    TESSERACT_CONFIG = '--psm 6'


class SensitiveInfo:
    """민감 정보 위치 정보"""
    def __init__(self, label: str, x: int, y: int, w: int, h: int, text: str = ""):
        self.label = label
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.text = text

    def __repr__(self):
        return f"SensitiveInfo({self.label}, {self.text}, x={self.x}, y={self.y})"


def is_date_pattern(text: str) -> bool:
    """
    날짜 패턴인지 확인 (YYYY-MM-DD 형태)

    Args:
        text: 검사할 텍스트

    Returns:
        날짜 패턴이면 True, 아니면 False
    """
    return bool(re.match(r'20\d{2}[-/]\d{2}[-/]\d{2}', text))


def extract_digits(text: str) -> str:
    """
    텍스트에서 숫자와 * 문자만 추출

    Args:
        text: 원본 텍스트

    Returns:
        숫자와 * 만 포함된 문자열
    """
    return re.sub(r'[^0-9*]', '', text)


def detect_sensitive_info(
    ocr_data: Dict[str, Any],
    min_confidence: int = OCRConfig.MIN_CONFIDENCE
) -> List[SensitiveInfo]:
    """
    OCR 데이터에서 민감 정보(카드번호, 8-10자리 숫자) 감지

    Args:
        ocr_data: pytesseract.image_to_data()의 출력 (DICT 형태)
        min_confidence: 최소 신뢰도 (기본값: 30)

    Returns:
        감지된 민감 정보 리스트
    """
    n_boxes = len(ocr_data['text'])
    positions = []

    for i in range(n_boxes):
        # 신뢰도가 낮으면 스킵
        if int(ocr_data['conf'][i]) < min_confidence:
            continue

        text = ocr_data['text'][i].strip()
        if not text:
            continue

        text_digits = extract_digits(text)

        # 카드번호 감지 (15자리 이상, * 포함)
        if len(text_digits) >= OCRConfig.CARD_NUMBER_MIN_LENGTH:
            x, y, w, h = (
                ocr_data['left'][i],
                ocr_data['top'][i],
                ocr_data['width'][i],
                ocr_data['height'][i]
            )
            positions.append(SensitiveInfo('카드번호', x, y, w, h, text))

        # 8~10자리 숫자 감지 (날짜 제외)
        elif (OCRConfig.SENSITIVE_NUMBER_MIN_LENGTH <= len(text_digits)
              <= OCRConfig.SENSITIVE_NUMBER_MAX_LENGTH):
            if not is_date_pattern(text):
                x, y, w, h = (
                    ocr_data['left'][i],
                    ocr_data['top'][i],
                    ocr_data['width'][i],
                    ocr_data['height'][i]
                )
                label = f'{len(text_digits)}자리'
                positions.append(SensitiveInfo(label, x, y, w, h, text))

    return positions


def apply_blur_to_image(
    img: Image.Image,
    sensitive_positions: List[SensitiveInfo],
    padding: int = OCRConfig.BLUR_PADDING,
    kernel_size: Tuple[int, int] = OCRConfig.BLUR_KERNEL_SIZE
) -> Image.Image:
    """
    이미지에 민감 정보 영역에 블러 처리 적용

    Args:
        img: 원본 PIL 이미지
        sensitive_positions: 블러 처리할 영역 리스트
        padding: 영역 주변 패딩 (기본값: 10)
        kernel_size: Gaussian blur 커널 크기 (기본값: (51, 51))

    Returns:
        블러 처리된 PIL 이미지
    """
    if not sensitive_positions:
        return img

    img_array = np.array(img)
    blurred = img_array.copy()

    for info in sensitive_positions:
        x1 = max(0, info.x - padding)
        y1 = max(0, info.y - padding)
        x2 = min(blurred.shape[1], info.x + info.w + padding)
        y2 = min(blurred.shape[0], info.y + info.h + padding)

        roi = blurred[y1:y2, x1:x2]
        if roi.size > 0:
            blurred_roi = cv2.GaussianBlur(roi, kernel_size, 0)
            blurred[y1:y2, x1:x2] = blurred_roi

    return Image.fromarray(blurred)


def process_image_with_ocr(
    img_path: str,
    lang: str = OCRConfig.TESSERACT_LANG,
    config: str = OCRConfig.TESSERACT_CONFIG
) -> Tuple[str, Dict[str, Any]]:
    """
    이미지에 OCR을 수행하여 텍스트와 위치 데이터 추출

    Args:
        img_path: 이미지 파일 경로
        lang: Tesseract 언어 (기본값: 'kor+eng')
        config: Tesseract 설정 (기본값: '--psm 6')

    Returns:
        (OCR 텍스트, OCR 데이터 딕셔너리) 튜플
    """
    img = Image.open(img_path)

    ocr_text = pytesseract.image_to_string(img, lang=lang, config=config)
    ocr_data = pytesseract.image_to_data(
        img,
        lang=lang,
        config=config,
        output_type=pytesseract.Output.DICT
    )

    return ocr_text, ocr_data


def extract_card_number_from_sensitive_info(sensitive_info: List[SensitiveInfo]) -> Optional[str]:
    """
    민감 정보 리스트에서 카드번호 추출

    Args:
        sensitive_info: 감지된 민감 정보 리스트

    Returns:
        카드번호 문자열 또는 None
    """
    for info in sensitive_info:
        if info.label == '카드번호':
            return info.text
    return None


def process_and_blur_image(img_path: str) -> Tuple[Image.Image, List[SensitiveInfo], str, Dict[str, Any]]:
    """
    이미지에서 OCR 수행, 민감 정보 감지, 블러 처리

    Args:
        img_path: 이미지 파일 경로

    Returns:
        (블러 처리된 이미지, 감지된 민감 정보 리스트, OCR 텍스트, OCR 데이터) 튜플
    """
    img = Image.open(img_path)

    # OCR 수행 (텍스트와 위치 데이터 모두 추출)
    ocr_text = pytesseract.image_to_string(
        img,
        lang=OCRConfig.TESSERACT_LANG,
        config=OCRConfig.TESSERACT_CONFIG
    )

    ocr_data = pytesseract.image_to_data(
        img,
        lang=OCRConfig.TESSERACT_LANG,
        config=OCRConfig.TESSERACT_CONFIG,
        output_type=pytesseract.Output.DICT
    )

    # 민감 정보 감지
    sensitive_positions = detect_sensitive_info(ocr_data)

    # 블러 처리 적용
    blurred_img = apply_blur_to_image(img, sensitive_positions)

    return blurred_img, sensitive_positions, ocr_text, ocr_data
