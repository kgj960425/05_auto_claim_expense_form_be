import pytesseract
from pdf2image import convert_from_path

# Tesseract 실행파일 경로
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# PDF 파일 경로
file_path = r"C:\Users\user\Desktop\target_folder\1.pdf"

# Poppler bin 경로 (bin 폴더 꼭 지정)
poppler_path = r"C:\Program Files\poppler-25.07.0\Library\bin"

# PDF → 이미지 변환
pages = convert_from_path(file_path, dpi=300, poppler_path=poppler_path)

all_text = []
for i, page in enumerate(pages, 1):
    print(f"Processing page {i}...")
    text = pytesseract.image_to_string(page, lang='kor+eng')
    all_text.append(f"=== Page {i} ===\n{text}")

# 결과 저장
with open('output.txt', 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(all_text))

print("완료!")
