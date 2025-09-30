import os,sys,pathlib,re
import fitz  # PyMuPDF
import pytesseract
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from pdf2image import convert_from_path

file_path = r"C:\Users\user\Desktop\target_folder\1.pdf"
poppler_path = r"C:\Program Files\poppler-25.07.0\Library\bin"
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


###### 1. pdf -> 이미지 변환
# PDF → 이미지 변환 (DPI 300 권장)
pages = convert_from_path(file_path, dpi=300, poppler_path=poppler_path)

####### 2. 이미지 전처리  (그레이 스케일, 테두리 제거)
# PIL → numpy array
img = np.array(pages[0])
cv2.imwrite("cropped_ocr_image1_pdf.png", img)
# 1.그레이 스케일
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite("cropped_ocr_image2_gray.png", gray)
# 2.이진화 (글자/잉크 강조)
# 가우시안 블러 (노이즈 제거 후 이진화 안정화)
blur = cv2.GaussianBlur(gray, (5,5), 0)

# Otsu 이진화 (자동 임계값)
_, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Adaptive Threshold (국소 영역별 이진화)
adaptive = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY, 35, 11
)

# 두 결과를 합쳐서 "강한 이진화" 효과
otsu_adaptive = cv2.bitwise_and(otsu, adaptive)
cv2.imwrite("cropped_ocr_image3_otsu_adaptive.png", otsu_adaptive)
# 3. 여백 제거(crop)
# 글자(흰색) 영역 좌표 찾기
coords = cv2.findNonZero(otsu_adaptive)
# 전체 글자 영역 bounding box 계산
x, y, w, h = cv2.boundingRect(coords)
# crop (여백 제거)
cropped = img[y:y+h, x:x+w]
cv2.imwrite("cropped_ocr_image4_cropped.png", cropped)

# 리사이즈, 작은 객체 무시, 라인 제거는 병2신이다.

# 4. OCR 실행
config = "--psm 6 --oem 3"
text = pytesseract.image_to_string(cropped, lang="kor", config=config)

# 5. 결과 출력
print("==== OCR 결과 ====")
print(text)
