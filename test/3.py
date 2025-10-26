import pytesseract
from pdf2image import convert_from_path

# Tesseract 경로 설정 (Windows의 경우)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

file_path = r"C:\Users\user\Desktop\target_folder\1.pdf"

# PDF를 이미지로 변환 후 OCR
pages = convert_from_path(file_path , dpi=300)

all_text = []
for i, page in enumerate(pages, 1):
    print(f"Processing page {i}...")
    text = pytesseract.image_to_string(page, lang='kor+eng')
    all_text.append(f"=== Page {i} ===\n{text}")

# 결과 저장
with open('output.txt', 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(all_text))

print("완료!")