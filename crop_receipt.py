import cv2
import numpy as np

def auto_crop_receipt(input_path, output_path="cropped_auto.png", show=False):
    """
    흰색 여백을 제거하고 본문(글자/색 영역)만 자동 crop.
    """
    # 한글 경로 호환 로드
    img_array = np.fromfile(input_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지를 불러오지 못했습니다.")

    # === 1️⃣ 전처리 (흑백 변환)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # === 2️⃣ 이진화 (흰 배경 제거용)
    # 흰색 배경은 255, 글자나 색 영역은 어두운 값
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # === 3️⃣ 노이즈 제거 및 덩어리 강화
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # === 4️⃣ 외곽선(Contour) 찾기
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("⚠️ 글자 영역을 찾지 못했습니다. 원본을 반환합니다.")
        cv2.imwrite(output_path, img)
        return img

    # === 5️⃣ 가장 큰 영역 선택
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)

    # === 6️⃣ 약간의 패딩 추가
    pad = 15
    x1, y1 = max(x - pad, 0), max(y - pad, 0)
    x2, y2 = min(x + w + pad, img.shape[1]), min(y + h + pad, img.shape[0])

    cropped = img[y1:y2, x1:x2]

    # === 7️⃣ 결과 저장
    cv2.imwrite(output_path, cropped)
    print(f"✅ 자동 크롭 완료 → {output_path}")

    if show:
        cv2.imshow("Cropped", cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return cropped