from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import tempfile
import os
import base64
import requests
import fitz
import pytesseract
import re
import pikepdf
import logging
from typing import Dict, Any, List

from utils.ocr_processor import process_and_blur_image, extract_card_number_from_sensitive_info
from utils.pdf_processor import process_pdf_for_ocr
from utils.receipt_extractor import extract_receipt_info
from utils.file_manager import (
    create_workspace,
    save_uploaded_pdf,
    generate_output_filename,
    cleanup_directory
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Tesseract 경로 설정 (환경별 자동 감지)
import platform
if platform.system() == 'Windows':
    # Windows 로컬 개발 환경
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    # CloudType 등 Linux 서버
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
# Linux/CloudType 배포 환경은 시스템에 설치된 tesseract 자동 사용 (설정 불필요)

app = FastAPI()

# Request Body 모델 정의
class MergeRequest(BaseModel):
    pdf_list: list[str]
    pdf_path: str
    output_file_name: str

@app.get("/serverCheck")
def custom_json():
    data = {"msg": "ok"}
    return JSONResponse(content=data, status_code=200)

@app.post("/ocr/upload")
async def blur_sensitive_info(files: List[UploadFile] = File(...), user_id: str = "tester") -> Dict[str, Any]:
    """
    PDF에서 민감 정보(카드번호, 8-10자리 숫자)를 자동으로 감지하고 블러 처리합니다.

    Args:
        files: 업로드된 PDF 파일 리스트 (다중 선택 가능, PDF만 허용)
        user_id: 사용자 ID (기본값: "tester")

    Returns:
        처리 결과 딕셔너리 (처리된 이미지 개수, 출력 폴더 등)

    Process:
        1. 업로드된 파일들 검증 (PDF만 허용)
        2. 작업 공간 생성 (PDF 저장 폴더 + OCR 결과 폴더)
        3. PDF 파일들 저장
        4. PDF에서 이미지 추출 (pymupdf4llm)
        5. 각 이미지에 OCR 수행 및 민감 정보 감지
        6. 감지된 영역에 블러 처리 적용
        7. 결과 이미지 저장

    Raises:
        HTTPException: PDF 파일이 없거나 처리 중 오류 발생 시
    """
    try:
        # 1. 파일 검증
        if not files:
            raise HTTPException(status_code=400, detail="파일이 업로드되지 않았습니다")

        # PDF 파일만 필터링
        pdf_files_to_upload = []
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                logger.warning(f"PDF가 아닌 파일 스킵: {file.filename}")
                continue
            pdf_files_to_upload.append(file)

        if not pdf_files_to_upload:
            raise HTTPException(status_code=400, detail="PDF 파일이 없습니다. PDF 파일만 업로드 가능합니다")

        logger.info(f"업로드된 PDF 파일: {len(pdf_files_to_upload)}개")

        # 2. 작업 공간 생성
        workspace = create_workspace(user_id)
        logger.info(f"작업 공간 생성 완료: {workspace.output_folder}")

        # 3. 각 PDF 파일 처리
        processed_count = 0
        temp_image_dirs_to_cleanup = []
        receipts_data = []  # 영수증 정보를 저장할 리스트

        for pdf_idx, file in enumerate(pdf_files_to_upload, 1):
            logger.info(f"[{pdf_idx}/{len(pdf_files_to_upload)}] 처리 중: {file.filename}")

            try:
                # PDF를 workspace.pdf_folder에 저장 (원본 보존)
                pdf_path = await save_uploaded_pdf(file, workspace.pdf_folder)
                pdf_name = os.path.splitext(file.filename)[0]

                # 임시 이미지 추출 폴더 (나중에 삭제할 것)
                temp_image_dir = tempfile.mkdtemp()
                temp_image_dirs_to_cleanup.append(temp_image_dir)

                # PDF에서 이미지 추출
                pdf_result = process_pdf_for_ocr(
                    pdf_path,
                    temp_image_dir,
                    verbose=False
                )

                logger.info(f"추출된 이미지: {len(pdf_result.extracted_images)}개")

                # 각 이미지에 OCR 및 블러 처리 적용
                for img_path in pdf_result.extracted_images:
                    try:
                        # 1. OCR 수행 및 민감 정보 감지 & 블러 처리 (한 번에)
                        blurred_img, sensitive_info, ocr_text, ocr_data = process_and_blur_image(img_path)

                        # 2. OCR 텍스트 + 데이터로부터 영수증 정보 추출
                        receipt_info = extract_receipt_info(ocr_text, ocr_data)

                        # 3. 블러 처리에서 감지한 카드번호 우선 사용 (더 정확함)
                        detected_card_number = extract_card_number_from_sensitive_info(sensitive_info)
                        if detected_card_number:
                            receipt_info.card_number = detected_card_number

                        # 4. 전체 작업 순서대로 번호 매기기 (processed_count + 1)
                        sequence_number = processed_count + 1

                        logger.info(f"  이미지 {sequence_number}: 영수증 정보 추출 완료")
                        logger.debug(f"    거래일자: {receipt_info.transaction_date}")
                        logger.debug(f"    카드번호: {receipt_info.card_number}")
                        logger.debug(f"    가맹점명: {receipt_info.merchant_name}")
                        logger.debug(f"    합계: {receipt_info.total_amount}")

                        # 5. 민감 정보 로깅
                        if sensitive_info:
                            logger.info(f"  이미지 {sequence_number}: 감지된 민감 정보 {len(sensitive_info)}개")
                            for info in sensitive_info:
                                logger.debug(f"    - {info.label}: {info.text}")

                        # 6. 블러 처리된 이미지 저장 (파일명_작업순서.png)
                        output_filename = generate_output_filename(pdf_name, sequence_number)
                        output_path = os.path.join(workspace.output_folder, output_filename)
                        blurred_img.save(output_path)

                        # 7. 영수증 정보와 파일명을 함께 저장
                        receipt_data = receipt_info.to_dict()
                        receipt_data['filename'] = output_filename
                        receipt_data['sequence_number'] = sequence_number
                        receipts_data.append(receipt_data)

                        processed_count += 1
                        logger.info(f"  저장 완료: {output_filename}")

                    except Exception as e:
                        logger.error(f"  이미지 처리 실패: {e}")
                        continue

                if not pdf_result.extracted_images:
                    logger.warning(f"PDF에서 이미지를 추출하지 못했습니다: {file.filename}")

            except Exception as e:
                logger.error(f"PDF 처리 실패 ({file.filename}): {e}")
                continue

        # 임시 이미지 디렉토리만 정리 (PDF 원본은 보존)
        for temp_dir in temp_image_dirs_to_cleanup:
            cleanup_directory(temp_dir)

        logger.info(f"임시 이미지 파일 정리 완료")

        # 7. 결과 반환
        return {
            "message": "success",
            "output_folder": workspace.output_folder,
            "receipts": receipts_data  # 영수증 정보 배열
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"블러 처리 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")

@app.post("/merge/")
async def merge_pdfs_in_order(request: MergeRequest):
    """
    사용자가 지정한 순서대로 PDF 파일을 병합합니다.
    {
        "pdf_list": [
            "0601_1.pdf",
            "0601_2.pdf",
            "0602_1.pdf",
            "0605_1.pdf",
            "0605_2.pdf",
            "0607_1.pdf",
            "0607_2.pdf",
            "0608_1.pdf",
            "0608_2.pdf",
            "0609_1.pdf",
            "0609_2.pdf",
            "0612_1.pdf",
            "0612_2.pdf",
            "0612_3.pdf",
            "0613_1.pdf",
            "0613_2.pdf",
            "0613_3.pdf",
            "0614_1.pdf",
            "0614_2.pdf",
            "0615_1.pdf",
            "0615_2.pdf",
            "0616_1.pdf",
            "0616_2.pdf",
            "0616_3.pdf",
            "0619_1.pdf",
            "0619_2.pdf",
            "0619_3.pdf",
            "0620_1.pdf",
            "0620_2.pdf",
            "0621_1.pdf",
            "0621_2.pdf",
            "0622_1.pdf",
            "0622_2.pdf",
            "0623_1.pdf",
            "0623_2(시건장치).pdf",
            "0626_1.pdf",
            "0627_1.pdf",
            "0627_2.pdf",
            "0628_1.pdf",
            "0630_1.pdf",
            "0630_2(인터넷랜어댑터).pdf"
        ],
        "pdf_path": "tester_20251027160000_cf1500be",
        "output_file_name": "merged_result.pdf"
    }
    """
    merged = pikepdf.Pdf.new()
    total_pages = 0
    base_pdf_path = f"static/temp/{request.pdf_path}/pdf"
    result_folder = f"static/temp/{request.pdf_path}/result"
    os.makedirs(result_folder, exist_ok=True)

    for pdf_filename in request.pdf_list:
        full_path = os.path.join(base_pdf_path, pdf_filename)
        try:
            src = pikepdf.Pdf.open(full_path)
            merged.pages.extend(src.pages)
            total_pages += len(src.pages)
            src.close()
        except Exception as e:
            print(f"{full_path} 병합 실패: {e}")

    # 병합된 파일을 result 폴더에 저장
    result_path = os.path.join(result_folder, request.output_file_name)
    merged.save(result_path)
    merged.close()
    print(f"완료: {result_path} (총 {total_pages} 페이지)")

    # 파일 다운로드 응답 반환
    return FileResponse(
        path=result_path,
        media_type="application/pdf",
        filename=request.output_file_name
    )