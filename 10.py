from pdf2image import convert_from_path
from crop_receipt import auto_crop_receipt

PDF_PATH = r"C:\Users\user\Desktop\expense_cliaim\1.pdf"
POPPLER_PATH = r"C:\Program Files\poppler-25.07.0\Library\bin"

# PDF → 이미지 변환
pages = convert_from_path(PDF_PATH, dpi=300, poppler_path=POPPLER_PATH)

# 첫 페이지를 PNG로 저장
temp_path = r"C:\Users\user\Desktop\expense_cliaim\page1.png"
pages[0].save(temp_path, "PNG")

# auto_crop_receipt() 호출
auto_crop_receipt(temp_path, r"C:\Users\user\Desktop\expense_cliaim\cropped_auto.png")
