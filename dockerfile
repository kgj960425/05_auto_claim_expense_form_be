FROM python:3.12-slim

# 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-kor \
    libtesseract-dev \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
 && apt-get clean

# 앱 복사
WORKDIR /app
COPY . .

# ✅ 전체 권한 부여 (쓰기 문제 방지)
RUN chmod -R 777 /app

# 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 실행 명령
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
