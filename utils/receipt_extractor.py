"""
영수증 정보 추출 유틸리티
"""
import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict


@dataclass
class ReceiptInfo:
    """영수증 정보"""
    transaction_date: Optional[str] = None  # 거래일자
    card_number: Optional[str] = None       # 카드번호
    merchant_name: Optional[str] = None     # 가맹점명
    total_amount: Optional[str] = None      # 합계

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)


def extract_transaction_date(text: str) -> Optional[str]:
    """
    거래일자 추출 (YYYY-MM-DD HH:MM:SS 또는 YYYY-MM-DD 형식)

    Args:
        text: OCR 텍스트

    Returns:
        거래일자 문자열 또는 None
    """
    # YYYY-MM-DD HH:MM:SS 형식
    pattern1 = r'\d{4}[-/.]\d{2}[-/.]\d{2}\s+\d{2}:\d{2}:\d{2}'
    match = re.search(pattern1, text)
    if match:
        return match.group(0)

    # YYYY-MM-DD 형식
    pattern2 = r'\d{4}[-/.]\d{2}[-/.]\d{2}'
    match = re.search(pattern2, text)
    if match:
        return match.group(0)

    # YYYY/MM/DD 형식도 허용
    pattern3 = r'\d{4}/\d{2}/\d{2}'
    match = re.search(pattern3, text)
    if match:
        return match.group(0).replace('/', '-')

    return None


def extract_card_number(text: str) -> Optional[str]:
    """
    카드번호 추출 (일부 마스킹된 형태 포함)

    Args:
        text: OCR 텍스트

    Returns:
        카드번호 문자열 또는 None
    """
    # 카드번호 패턴 (숫자와 * 혼합, 하이픈/공백 포함 가능)
    # 예: 1234-****-****-5678, 1234 **** **** 5678
    patterns = [
        r'\d{4}[-\s*]+\d{4}[-\s*]+\d{4}[-\s*]+\d{4}',  # 16자리
        r'\d{4}[-\s*]+\*{4}[-\s*]+\*{4}[-\s*]+\d{4}',  # 일부 마스킹
        r'\d{4}[-\s]+\d{4}[-\s]+\d{4}[-\s]+\d{4}',     # 하이픈 또는 공백
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)

    return None


def extract_merchant_name(text: str) -> Optional[str]:
    """
    가맹점명 추출 (텍스트 기반)

    Args:
        text: OCR 텍스트

    Returns:
        가맹점명 문자열 또는 None
    """
    # "가맹점" 키워드 다음의 텍스트 추출
    patterns = [
        r'가맹점[:\s]*([^\n]+)',
        r'상호[:\s]*([^\n]+)',
        r'매장[:\s]*([^\n]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            merchant = match.group(1).strip()
            # 특수문자 제거 및 정리
            merchant = re.sub(r'[:\-=]+', '', merchant).strip()
            if merchant:
                return merchant

    # 가맹점 정보가 없으면 첫 줄을 가맹점명으로 추정 (옵션)
    lines = text.split('\n')
    for line in lines[:5]:  # 상위 5줄만 확인
        line = line.strip()
        if len(line) > 2 and not re.match(r'^\d', line):
            return line

    return None


def find_nearby_text(
    ocr_data: Dict[str, Any],
    target_index: int,
    min_confidence: int = 30,
    max_lines: int = 3
) -> List[str]:
    """
    특정 텍스트 근처의 텍스트 찾기 (우측 + 하단 여러 줄)

    Args:
        ocr_data: OCR 데이터
        target_index: 타겟 텍스트의 인덱스
        min_confidence: 최소 신뢰도
        max_lines: 최대 확인할 줄 수

    Returns:
        근처 텍스트 리스트
    """
    n_boxes = len(ocr_data['text'])
    nearby_texts = []

    current_y = ocr_data['top'][target_index]
    current_x = ocr_data['left'][target_index] + ocr_data['width'][target_index]
    lines_found = 0

    # 근처 20개 박스 확인 (더 넓게)
    for j in range(target_index + 1, min(target_index + 20, n_boxes)):
        if int(ocr_data['conf'][j]) < min_confidence:
            continue

        next_text = ocr_data['text'][j].strip()
        if not next_text or next_text in [':', '-', '=']:
            continue

        next_y = ocr_data['top'][j]
        next_x = ocr_data['left'][j]

        # 같은 줄 우측 (y 좌표 차이 20 이하, x가 더 큼)
        if abs(next_y - current_y) <= 20 and next_x >= current_x:
            nearby_texts.append(next_text)
        # 바로 아래 (y 좌표 차이 20~80 사이)
        elif 20 < (next_y - current_y) <= 80:
            nearby_texts.append(next_text)
            lines_found += 1
            current_y = next_y  # 다음 줄로 이동

            if lines_found >= max_lines:
                break

    return nearby_texts


def extract_merchant_name_from_ocr_data(
    ocr_data: Dict[str, Any],
    min_confidence: int = 30
) -> Optional[str]:
    """
    가맹점명 추출 (OCR 데이터 기반, 위치 정보 활용)

    "가맹점명" 텍스트의 위치를 찾아서 그 근처(우측 또는 바로 아래)의 텍스트를 추출

    Args:
        ocr_data: pytesseract.image_to_data()의 출력 (DICT 형태)
        min_confidence: 최소 신뢰도 (기본값: 30)

    Returns:
        가맹점명 문자열 또는 None
    """
    n_boxes = len(ocr_data['text'])

    # 1. "가맹점명" 키워드의 위치 찾기
    merchant_label_index = -1
    merchant_label_y = -1
    merchant_label_x_end = -1
    merchant_label_x_start = -1

    for i in range(n_boxes):
        if int(ocr_data['conf'][i]) < min_confidence:
            continue

        text = ocr_data['text'][i].strip()

        # "가맹점명" 텍스트 찾기 (최소 3글자 이상 + 키워드 포함)
        if len(text) >= 3 and ('가맹점' in text or text in ['가맹점명', '상호명', '매장명']):
            merchant_label_index = i
            merchant_label_y = ocr_data['top'][i]
            merchant_label_x_start = ocr_data['left'][i]
            merchant_label_x_end = ocr_data['left'][i] + ocr_data['width'][i]
            break

    # "가맹점명" 텍스트를 못 찾으면 다른 방법 시도
    if merchant_label_index == -1:
        return None

    # 2. "가맹점명" 위치 기준으로 y축 아래 + x축 오른쪽만 탐색
    merchant_texts = []

    for i in range(merchant_label_index + 1, n_boxes):
        if int(ocr_data['conf'][i]) < min_confidence:
            continue

        text = ocr_data['text'][i].strip()

        # 빈 텍스트나 한 글자는 제외
        if not text or len(text) <= 1:
            continue

        # 구분자 제외
        if text in [':', '-', '=', '|', ':', '명']:
            continue

        # '가맹점', '상호', '매장' 등 키워드가 포함된 텍스트는 무조건 제외
        if '가맹점' in text or '상호' in text or '매장' in text or '명' == text:
            continue

        y = ocr_data['top'][i]
        x = ocr_data['left'][i]

        # y축: 아래만 (같은 줄 포함, ±15px 이내)
        # x축: 오른쪽만 (label 끝보다 오른쪽)
        if abs(y - merchant_label_y) <= 15 and x >= merchant_label_x_end:
            merchant_texts.append(text)
        # y축: 아래 (15~100px)
        # x축: 오른쪽 (label 끝보다 오른쪽)
        elif 15 < (y - merchant_label_y) <= 100 and x >= merchant_label_x_end:
            merchant_texts.append(text)

        # 텍스트 3개 이상 수집하면 종료
        if len(merchant_texts) >= 3:
            break

    if merchant_texts:
        # 공백 없이 결합
        merchant_name = ''.join(merchant_texts)
        return merchant_name if len(merchant_name) > 2 else None

    # '가맹점명' 키워드를 찾았는데 오른쪽/아래에 텍스트가 없으면 None 리턴
    # (fallback 로직 실행하지 않음)
    return None

    # 2. 키워드가 없으면 상단의 텍스트를 가맹점명으로 추정 (사용 안함)
    if n_boxes > 0:
        max_y = max(ocr_data['top'][i] for i in range(n_boxes) if int(ocr_data['conf'][i]) >= min_confidence)
        top_20_percent = max_y * 0.2  # 30%에서 20%로 축소 (더 상단만 확인)

        # 상단 영역에서 텍스트들 수집 (여러 줄 결합)
        merchant_parts = []

        for i in range(n_boxes):
            if int(ocr_data['conf'][i]) < min_confidence:
                continue

            text = ocr_data['text'][i].strip()
            y = ocr_data['top'][i]

            # 상단 20% 영역, 2자 이상, 숫자로 시작하지 않음, 특수문자 제외
            if (y <= top_20_percent and
                len(text) >= 2 and
                not re.match(r'^\d', text) and
                text not in [':', '-', '=', '|', '/', '\\']):
                merchant_parts.append({'text': text, 'y': y, 'index': i})

        if merchant_parts:
            # y 좌표로 정렬 (상단부터)
            merchant_parts.sort(key=lambda x: x['y'])

            # 연속된 줄 찾기 (y 좌표 차이가 60 이하인 것들)
            current_line = [merchant_parts[0]['text']]
            current_y = merchant_parts[0]['y']

            for i in range(1, min(len(merchant_parts), 5)):  # 최대 5줄까지만
                next_y = merchant_parts[i]['y']
                text = merchant_parts[i]['text']

                # 같은 줄이거나 다음 줄 (y 차이 60 이하)
                if abs(next_y - current_y) <= 60:
                    current_line.append(text)
                    current_y = next_y
                else:
                    break

            merchant_name = ' '.join(current_line)

            # 여전히 짧으면 (5자 이하) 더 추가
            if len(merchant_name) <= 5 and merchant_parts:
                first_index = merchant_parts[0]['index']
                nearby_texts = find_nearby_text(ocr_data, first_index, min_confidence)
                if nearby_texts:
                    merchant_name = merchant_name + ' ' + ' '.join(nearby_texts)

            return merchant_name.strip()

    return None


def extract_total_amount(text: str) -> Optional[str]:
    """
    합계 금액 추출

    Args:
        text: OCR 텍스트

    Returns:
        금액 문자열 (콤마 포함) 또는 None
    """
    # "합계", "총액", "결제금액" 등의 키워드 근처에서 금액 찾기
    patterns = [
        r'합계[:\s]*(\d{1,3}(?:,\d{3})*)\s*원',
        r'총액[:\s]*(\d{1,3}(?:,\d{3})*)\s*원',
        r'결제금액[:\s]*(\d{1,3}(?:,\d{3})*)\s*원',
        r'금액[:\s]*(\d{1,3}(?:,\d{3})*)\s*원',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)

    # 키워드가 없으면 가장 큰 금액을 찾기
    all_amounts = re.findall(r'(\d{1,3}(?:,\d{3})+)\s*원', text)
    if all_amounts:
        # 콤마 제거 후 정수 변환하여 가장 큰 금액 찾기
        max_amount = max(all_amounts, key=lambda x: int(x.replace(',', '')))
        return max_amount

    return None


def extract_receipt_info(text: str, ocr_data: Optional[Dict[str, Any]] = None) -> ReceiptInfo:
    """
    OCR 텍스트에서 영수증 정보 추출

    Args:
        text: OCR 텍스트
        ocr_data: OCR 데이터 (위치 정보 포함, 선택사항)

    Returns:
        ReceiptInfo 객체 (거래일자, 카드번호, 가맹점명, 합계)
    """
    # 가맹점명 추출 - OCR 데이터가 있으면 우선 사용
    merchant_name = None
    if ocr_data:
        merchant_name = extract_merchant_name_from_ocr_data(ocr_data)

    # OCR 데이터에서 추출 실패하면 텍스트 기반 추출
    if not merchant_name:
        merchant_name = extract_merchant_name(text)

    receipt = ReceiptInfo(
        transaction_date=extract_transaction_date(text),
        card_number=extract_card_number(text),
        merchant_name=merchant_name,
        total_amount=extract_total_amount(text)
    )

    return receipt
