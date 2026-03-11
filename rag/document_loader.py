"""
PDF 문서 로딩 및 파싱 모듈

실무 포인트:
RAG 시스템의 품질은 문서 로딩 단계에서부터 결정된다.
텍스트 추출이 제대로 안 되면 아무리 좋은 검색 알고리즘도 소용없다.

pypdf 선택 이유:
- 순수 Python, 설치 간단
- 대부분의 PDF 처리 가능
- 실무에서 가장 많이 쓰이는 PDF 라이브러리 중 하나

한계:
- 스캔된 PDF (이미지 기반) → OCR 필요 (pytesseract 등)
- 복잡한 표/레이아웃 → 파싱 품질 저하 가능
"""

from pathlib import Path
from dataclasses import dataclass
from pypdf import PdfReader


@dataclass
class Document:
    """
    파싱된 문서 단위 데이터 클래스

    실무 포인트:
    page_num, source 같은 메타데이터가 매우 중요하다.
    나중에 "몇 페이지에서 찾았는지" 출처를 보여줄 때 사용한다.
    """
    content: str        # 텍스트 내용
    page_num: int       # 페이지 번호
    source: str         # 파일 경로 (출처 표시용)
    metadata: dict      # 추가 메타데이터


class DocumentLoader:
    """
    PDF 문서 로더 클래스

    실무 포인트:
    지금은 PDF만 지원하지만
    실제 서비스에서는 Word, Excel, 웹페이지 등
    다양한 소스를 지원해야 한다.
    load() 인터페이스를 통일해두면
    나중에 다른 형식을 추가할 때 쉽게 확장 가능하다.
    """

    def load_pdf(self, file_path: str) -> list[Document]:
        """
        PDF 파일을 로딩해서 페이지별 Document 리스트 반환

        Args:
            file_path: PDF 파일 경로

        Returns:
            페이지별 Document 객체 리스트
        """

        path = Path(file_path)

        # 파일 존재 여부 확인 (Fail Fast)
        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        if path.suffix.lower() != ".pdf":
            raise ValueError(f"PDF 파일만 지원합니다: {file_path}")

        print(f"📄 PDF 로딩 중: {path.name}")

        documents = []
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)

        for page_num, page in enumerate(reader.pages, start=1):
            # 텍스트 추출
            text = page.extract_text()

            # 빈 페이지 스킵
            if not text or not text.strip():
                print(f"   ⚠️  {page_num}페이지: 텍스트 없음 (스킵)")
                continue

            # 텍스트 정제
            text = self._clean_text(text)

            documents.append(Document(
                content=text,
                page_num=page_num,
                source=str(path.name),
                metadata={
                    "file_path": str(path),
                    "total_pages": total_pages,
                }
            ))

            print(f"   ✅ {page_num}/{total_pages} 페이지 로딩 완료"
                  f" ({len(text)}자)")

        print(f"\n📚 총 {len(documents)}페이지 로딩 완료\n")
        return documents

    def _clean_text(self, text: str) -> str:
        """
        추출된 텍스트 정제

        실무 포인트:
        PDF에서 추출한 텍스트는 불필요한 공백, 줄바꿈이 많다.
        정제를 잘 해야 청킹 품질이 올라간다.
        """

        # 연속 공백 → 단일 공백
        import re
        text = re.sub(r" {2,}", " ", text)

        # 3개 이상 연속 줄바꿈 → 2개로 정규화
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()