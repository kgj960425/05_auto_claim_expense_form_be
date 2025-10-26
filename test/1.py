import fitz
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

file_path = r"C:\Users\user\Desktop\target_folder\1.pdf"

doc = fitz.open(file_path)
page = doc[0]  # 첫 페이지

card_rect  = fitz.Rect(80, 160, 500, 220)   # 카드번호 (상단부)
date_rect  = fitz.Rect(80, 260, 500, 320)   # 거래일자 (카드번호 아래)
store_rect = fitz.Rect(80, 380, 550, 440)   # 가맹점명 (중간부)
total_rect = fitz.Rect(300, 700, 560, 780)  # 합계 (하단부)

def ocr_region(page, rect, lang="kor+eng"):
    pix = page.get_pixmap(dpi=300, clip=card_rect)
    pix.save("debug_card.png")

    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    text = pytesseract.image_to_string(img, lang=lang, config="--psm 6 --oem 3")
    return text.strip()

print("거래일자:", ocr_region(page, date_rect))
print("카드번호:", ocr_region(page, card_rect, lang="eng"))
print("합계:", ocr_region(page, total_rect))
print("가맹점명:", ocr_region(page, store_rect, lang="kor"))
