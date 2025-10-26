from google import genai

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain how AI works in a few words",
)

print(response.text)

# 1. 전체 읽기.
# 2. crop 설정으로 자르기
# 3. poppler 로 이미지로 파싱
# 4. pymupdf4llm 로 ocr
# 5. react spreadjs로 엑셀로 변환