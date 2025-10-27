"""
PDF 처리 및 이미지 추출 유틸리티
"""
import os
import fitz
import pymupdf4llm
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PDFMetadata:
    """PDF 메타데이터"""
    total_pages: int
    has_text: bool
    file_name: str
    file_size: int


@dataclass
class PDFProcessResult:
    """PDF 처리 결과"""
    pdf_path: str
    extracted_images: List[str]
    markdown_text: str
    metadata: PDFMetadata


def check_pdf_text_content(pdf_path: str, verbose: bool = False) -> Dict[str, Any]:
    """
    PDF에 텍스트가 있는지 확인

    Args:
        pdf_path: PDF 파일 경로
        verbose: 상세 정보 출력 여부

    Returns:
        페이지별 텍스트 정보 딕셔너리
    """
    doc = fitz.open(pdf_path)
    page_info = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        page_info.append({
            'page_num': page_num + 1,
            'text_length': len(text),
            'has_text': bool(text.strip()),
            'preview': text[:200] if text.strip() else None
        })

        if verbose:
            print(f"\nPage {page_num + 1} - Text length: {len(text)}")
            if text.strip():
                print(f"First 200 chars: {text[:200]}")
            else:
                print("⚠️ No text found - This is likely an image-based PDF")

    doc.close()

    return {
        'total_pages': len(page_info),
        'has_text': any(p['has_text'] for p in page_info),
        'pages': page_info
    }


def extract_images_from_pdf(
    pdf_path: str,
    output_dir: str,
    verbose: bool = False
) -> List[str]:
    """
    PDF에서 이미지 추출 (pymupdf4llm 사용)

    Args:
        pdf_path: PDF 파일 경로
        output_dir: 이미지 저장 디렉토리
        verbose: 상세 정보 출력 여부

    Returns:
        추출된 이미지 파일 경로 리스트
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        md_with_images = pymupdf4llm.to_markdown(
            pdf_path,
            write_images=True,
            image_path=output_dir,
            page_chunks=True
        )

        if verbose:
            chunks_count = len(md_with_images) if isinstance(md_with_images, list) else 'N/A'
            print(f"   Success! Chunks: {chunks_count}")
            if isinstance(md_with_images, list) and md_with_images:
                print(f"   First chunk: {md_with_images[0].get('text', '')[:300]}")

    except Exception as e:
        if verbose:
            print(f"   Error extracting images: {e}")
        raise

    # 추출된 이미지 파일 목록 반환
    if os.path.exists(output_dir):
        image_files = [
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if f.endswith('.png')
        ]
        return sorted(image_files)

    return []


def pdf_to_markdown(pdf_path: str, page_chunks: bool = False) -> str:
    """
    PDF를 마크다운 형식으로 변환

    Args:
        pdf_path: PDF 파일 경로
        page_chunks: 페이지별로 분할할지 여부

    Returns:
        마크다운 텍스트 또는 청크 리스트
    """
    if page_chunks:
        return pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    else:
        return pymupdf4llm.to_markdown(pdf_path)


def get_pdf_metadata(pdf_path: str) -> PDFMetadata:
    """
    PDF 메타데이터 추출

    Args:
        pdf_path: PDF 파일 경로

    Returns:
        PDFMetadata 객체
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    # 텍스트가 있는지 확인
    has_text = False
    for page_num in range(total_pages):
        page = doc[page_num]
        if page.get_text().strip():
            has_text = True
            break

    doc.close()

    return PDFMetadata(
        total_pages=total_pages,
        has_text=has_text,
        file_name=os.path.basename(pdf_path),
        file_size=os.path.getsize(pdf_path)
    )


def process_pdf_for_ocr(
    pdf_path: str,
    output_dir: str,
    verbose: bool = False
) -> PDFProcessResult:
    """
    PDF를 OCR 처리를 위해 전처리

    Args:
        pdf_path: PDF 파일 경로
        output_dir: 출력 디렉토리
        verbose: 상세 정보 출력 여부

    Returns:
        PDFProcessResult 객체
    """
    # 메타데이터 추출
    metadata = get_pdf_metadata(pdf_path)

    if verbose:
        print(f"\n{'='*80}")
        print(f"Processing: {metadata.file_name}")
        print(f"Total pages: {metadata.total_pages}")
        print(f"Has text: {metadata.has_text}")
        print(f"{'='*80}")

    # 텍스트 내용 확인
    if verbose:
        check_pdf_text_content(pdf_path, verbose=True)

    # 마크다운 변환
    md_text = pdf_to_markdown(pdf_path)

    # 이미지 추출
    pdf_name = os.path.splitext(metadata.file_name)[0]
    temp_image_dir = os.path.join(output_dir, pdf_name)
    extracted_images = extract_images_from_pdf(pdf_path, temp_image_dir, verbose)

    return PDFProcessResult(
        pdf_path=pdf_path,
        extracted_images=extracted_images,
        markdown_text=md_text,
        metadata=metadata
    )
