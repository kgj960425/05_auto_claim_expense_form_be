## setting

- git init
- python -m venv .venv
- .\.venv\Scripts\Activate.ps1
- pip install -r requirements.txt
- uvicorn main:app --host 0.0.0.0 --port 8000 --reload


build
# testfile이라는 이름으로 빌드
docker build -t testfile:latest .

# 빌드 완료 후 확인
docker images | findstr testfile

# tar 파일로 저장
docker save -o testfile.tar testfile:latest

# 파일 생성 확인
dir testfile.tar

# testfile.tar 파일이 있는 위치로 이동
cd ~

# 기존 컨테이너 중지 및 삭제
docker stop testfile
docker rm testfile

# 기존 testfile 이미지 삭제
docker rmi testfile:latest

# Docker 이미지 로드
docker load -i testfile.tar

# 로드 확인
docker images

# 컨테이너 실행
docker run -d \
  --name testfile \
  -p 8000:8000 \
  -v ~/testfile/static:/app/static \
  -v ~/testfile/temp:/app/temp \
  --env-file ~/testfile/.env \
  --restart unless-stopped \
  testfile:latest

# 컨테이너 상태 확인
docker ps

# 로그 확인
docker logs -f testfile

# API 테스트
curl http://localhost:8000/serverCheck





# CloudType으로 배포 할 때 고려해야 할 것.
# 1. requirement.txt - 지원하지 않는 라이브러리 업데이트를 위해 cloudType.yaml에 apt-get update 필요
# 2. api 서버의 경우 Health Check의 api 호출이 되는지 여부로 프로젝트 배포 여부를 확인함. 배포 port 확인 잘 할 것.
# 3. .env는 프로젝트 배포 설정시 Environment variables에 작성.




# ocr 전처리 방법 확인되는대로 넣었음.
# 1. 그레이 스케일 컬러 -> 측백으로 바꿔서
# 2. 노이즈 제거
# 3. 이진화 (바이너리)
# 4. 팽창과 침식 (글자 연결, 글자 분리)
# 5. 윤곽선 검출 (contours)
# 6. 모폴로지 연산 (morphology)
# 7. 기울기 보정 (deskew)
# 8. 경계 상자 (bounding box)
# 9. 원근 변환 (perspective transform)
# 10. 히스토그램 평활화 (histogram equalization)
# 11. 이미지 필터링 (Gaussian, Median, Bilateral)
# 12. 이미지 크기 조정 (scaling)
# 13. 이미지 회전 (rotation)
# 14. 이미지 자르기 (cropping)
# 15. 이미지 평활화 (smoothing)
# 16. 대비 향상 (contrast enhancement)
# 17. 엣지 검출 (edge detection)
# 18. 컬러 공간 변환 (color space conversion)
# 19. 템플릿 매칭 (template matching)
# 20. 딥러닝 기반 전처리 (Deep learning based preprocessing)
# 21. 배경 제거 (background removal)
# 22.  JSON Schema로 explanation이랑 enum 리스트 던져주고 고르게 하는 방법
# OpenAI API를 통해 모델 호출하면 모델에 JSON 포맷으로 응답하는걸 강제할 수 있는데,
# 여기서 시스템 프롬프트와는 별개로 모델한테 선택을 강제할 수 있습니다. Key-Value 값에서 Value 값중에 고를 수 있는 선택지를 강제할 수도 있구요. (물론 P값에 따라 헛소리를 아주 가끔 하긴 하지만 맥락은 일치합니다)
# 그리고 시스템 프롬프트는 길어지고 명령간의 논리이해관계가 복잡해질수록 실수와 명령 불이행이 잦아지는데, JSON Schema 내부에서 Description으로 각 선택지에 대한 설명을 추가로 제공하면 이해도와 수행도가 훨씬 높아집니다. 모델들이 보통 벤치마크 뻥튀기를 위해 coding 능력과 수학 능력을 위주로 학습하는 만큼 오히려 이쪽이 긴 자연어 뭉치보다 더 잘 따라오는 것 같기도 합니다.
# OpenAI-like API 구조
# OCR은 다른 걸로 처리하고 마크다운이랑 bounding boxes 값, 그리고 문자열 추출 한 다음 그걸 Qwen 4B로 처리해서 원하는 정보를 추출