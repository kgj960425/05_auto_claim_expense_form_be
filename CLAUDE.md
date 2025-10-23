# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

PDF 영수증에서 구조화된 데이터를 추출하는 OCR 기반 경비 청구서 처리 시스템입니다. Tesseract OCR과 OpenCV 전처리를 사용하여 한국어/영어 영수증 PDF에서 거래 내역(날짜, 금액, 가맹점명)을 추출하고 CSV 형식으로 출력합니다.

## 환경 설정

### 초기 설정
```bash
# 가상환경 생성
python -m venv .venv

# 가상환경 활성화 (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# 의존성 설치
pip install -r requirements.txt
```

### 외부 의존성
- **Tesseract OCR**: `C:\Program Files\Tesseract-OCR\tesseract.exe`에 설치되어 있어야 함 (Windows)
- **Poppler**: pdf2image에 필요, 일반적으로 `C:\Program Files\poppler-25.07.0\Library\bin`에 위치
- 두 경로 모두 스크립트에 하드코딩되어 있으며 환경에 따라 수정이 필요할 수 있음

### 환경 변수
- 루트 디렉토리에 `.env` 파일 생성
- 필수: `GOOGLE_API_KEY` (google_ai_studio.py에서 Google Gemini API 사용)

## 코드 구조 및 아키텍처

### 메인 처리 스크립트

**app.py** - 프로덕션 배치 프로세서
- `~/Desktop/target_folder`의 모든 PDF 파일 처리
- PyMuPDF (fitz)를 사용하여 PDF를 이미지로 변환
- Adaptive Threshold 전처리 적용
- 추출 항목: 거래일자(YYYY-MM-DD HH:MM:SS), 금액(콤마 포함), 가맹점명
- 출력: `~/Desktop/expense_result.csv`

**7.py** - 고급 전처리 파이프라인
- 포괄적인 OCR 전처리 기법 구현
- pdf2image로 300 DPI 변환
- 전처리 단계: 그레이스케일 → 기울기 보정(각도 감지) → Otsu 이진화
- 결과를 `{pdf_name}_result.txt`로 저장

**6.py** - 이중 임계값 전처리 방식
- Otsu와 Adaptive Threshold를 결합하여 강건한 이진화 수행
- 자동 여백 제거(cropping) 구현
- cv2.findNonZero + boundingRect로 콘텐츠 영역 감지

### 실험용 스크립트 (1.py - 5.py)

**1.py** - 영역 기반 OCR (좌표 클리핑)
- fitz.Rect로 특정 영역 정의 (카드번호, 날짜, 가맹점, 합계)
- 일관된 레이아웃의 영수증에 유용

**기타 번호 스크립트** - 다양한 OpenCV 기법을 테스트하는 OCR 전처리 실험들

### Google Gemini 통합

**google_ai_studio.py** - AI 기반 데이터 추출
- Gemini 2.5 Flash 모델을 사용한 지능형 텍스트 해석
- 두 가지 API 방식 시연: google.generativeai와 google.genai.Client
- JSON Schema를 사용한 구조화된 데이터 추출의 향후 통합 지점 (readme 참고)

## OCR 전처리 전략

프로젝트에서 탐색하는 22가지 이상의 전처리 기법 (readme.md 참고):
1. 그레이스케일 변환
2. 노이즈 제거
3. 이진화 (Otsu, Adaptive Threshold)
4. 모폴로지 연산 (팽창, 침식)
5. 윤곽선 검출
6. 기울기 보정
7. 경계 상자 검출
8. 원근 변환
9. 히스토그램 평활화
10. 이미지 필터링 (Gaussian, Median, Bilateral)
11. 크기 조정/회전/자르기
12. 대비 향상
13. 엣지 검출
14. 템플릿 매칭
15. 배경 제거

**현재 권장 방식** (app.py와 7.py 기준):
- 300 DPI 변환
- Gaussian blur를 적용한 Adaptive Threshold
- cv2.minAreaRect를 통한 각도 보정
- Tesseract의 PSM 3/6과 OEM 3 사용

## 코드 실행 방법

### PDF OCR 처리
```bash
# Desktop/target_folder의 모든 PDF 배치 처리
python app.py

# 기울기 보정을 포함한 고급 전처리
python 7.py

# 이중 임계값 방식 + 자동 여백 제거
python 6.py
```

### Google Gemini 통합 테스트
```bash
python google_ai_studio.py
```

## 개발 참고사항

### Tesseract 설정
- `--psm 6`: 균일한 텍스트 블록 가정 (영수증용)
- `--psm 3`: 자동 페이지 분할 (전체 문서용)
- `--oem 3`: 기본 LSTM 신경망 모드
- `lang="kor+eng"`: 한국어 + 영어 언어 모델

### 데이터 추출 패턴 (app.py)
- **날짜**: `\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}`
- **금액**: `(\d{1,3}(?:,\d{3})*)\s*원`
- **가맹점**: "가맹점" 키워드 이후 텍스트

### 향후 통합 계획 (readme.md 참고)
정규식 추출을 Gemini API + JSON Schema로 대체:
- 구조화된 출력 모드로 필드 선택 강제
- 범주형 데이터에 enum 제약 추가
- JSON Schema의 필드 설명으로 모델 이해도 향상
- 처리 흐름: OCR 텍스트 → 마크다운 + bounding boxes → Qwen 4B → 구조화된 출력