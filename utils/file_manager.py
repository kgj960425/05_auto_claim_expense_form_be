"""
파일 및 디렉토리 관리 유틸리티
"""
import os
import glob
import uuid
from datetime import datetime
from typing import List, Tuple, Optional
from dataclasses import dataclass
from fastapi import UploadFile


@dataclass
class WorkspaceInfo:
    """작업 공간 정보"""
    pdf_folder: str
    output_folder: str
    timestamp: str
    job_uuid: str
    user_id: str


def create_workspace(
    user_id: str,
    base_dir: str = "static/temp"
) -> WorkspaceInfo:
    """
    OCR 처리를 위한 작업 공간 생성

    Args:
        user_id: 사용자 ID
        base_dir: 기본 디렉토리 (기본값: "static/temp")

    Returns:
        WorkspaceInfo 객체
    """
    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d%H%M%S")
    job_uuid = str(uuid.uuid4())[:8]

    # PDF 저장 폴더
    pdf_folder = os.path.join(base_dir, f"{user_id}_{timestamp}_{job_uuid}", "pdf")
    os.makedirs(pdf_folder, exist_ok=True)

    # OCR 결과 저장 폴더
    output_folder = os.path.join(base_dir, f"{user_id}_{timestamp}_{job_uuid}", "ocr")
    os.makedirs(output_folder, exist_ok=True)

    return WorkspaceInfo(
        pdf_folder=pdf_folder,
        output_folder=output_folder,
        timestamp=timestamp,
        job_uuid=job_uuid,
        user_id=user_id
    )


async def save_uploaded_pdf(
    file: UploadFile,
    target_dir: str
) -> str:
    """
    업로드된 PDF 파일을 원본 파일명으로 저장

    Args:
        file: FastAPI UploadFile 객체
        target_dir: 저장할 디렉토리

    Returns:
        저장된 파일의 절대 경로
    """
    os.makedirs(target_dir, exist_ok=True)

    # 원본 파일명 사용
    file_path = os.path.join(target_dir, file.filename)

    content = await file.read()
    with open(file_path, 'wb') as f:
        f.write(content)

    return file_path


def find_pdf_files(directory: str) -> List[str]:
    """
    디렉토리 내 모든 PDF 파일 찾기

    Args:
        directory: 검색할 디렉토리

    Returns:
        PDF 파일 경로 리스트
    """
    pattern = os.path.join(directory, "*.pdf")
    pdf_files = glob.glob(pattern)
    return sorted(pdf_files)


def generate_output_filename(
    pdf_name: str,
    page_number: int
) -> str:
    """
    출력 파일명 생성 (간단한 형식: 원본명_01.png)

    Args:
        pdf_name: PDF 파일명 (확장자 제외)
        page_number: 페이지 번호 (1부터 시작)

    Returns:
        생성된 파일명 (예: receipt_01.png)
    """
    return f"{pdf_name}_{page_number:02d}.png"


def cleanup_temp_files(*file_paths: str) -> None:
    """
    임시 파일 정리

    Args:
        *file_paths: 삭제할 파일 경로들
    """
    for file_path in file_paths:
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Warning: Failed to delete {file_path}: {e}")


def cleanup_directory(directory: str) -> None:
    """
    디렉토리와 내부 모든 파일 삭제

    Args:
        directory: 삭제할 디렉토리 경로
    """
    import shutil
    try:
        if directory and os.path.exists(directory):
            shutil.rmtree(directory)
    except Exception as e:
        print(f"Warning: Failed to delete directory {directory}: {e}")


def get_image_files(directory: str) -> List[str]:
    """
    디렉토리 내 모든 PNG 이미지 파일 찾기

    Args:
        directory: 검색할 디렉토리

    Returns:
        이미지 파일 경로 리스트
    """
    if not os.path.exists(directory):
        return []

    image_files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith('.png')
    ]
    return sorted(image_files)
