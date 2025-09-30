import fitz

doc = fitz.open(r"C:\Users\user\Desktop\target_folder\1.pdf")
page = doc[0]
print("페이지 크기:", page.rect)