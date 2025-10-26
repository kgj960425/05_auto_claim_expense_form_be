import google.generativeai as genai
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# os.getenv()를 사용하여 환경 변수에서 API 키를 가져옵니다.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# 사용할 모델을 선택합니다.
model = genai.GenerativeModel('gemini-2.5-flash')

# 모델에 전송할 프롬프트를 입력합니다.
response = model.generate_content("태양계 행성의 종류에 대하여 알려줘")

# 생성된 텍스트를 출력합니다.
print(response.text)
print(response)