"""
문서 청킹(분할) 모듈

실무 포인트:
청킹 전략은 RAG 성능에 가장 큰 영향을 미치는 요소 중 하나다.
chunk_size와 chunk_overlap을 잘못 설정하면
검색 품질이 크게 떨어진다.

청킹 전략 종류:
1. Fixed-size Chunking   → 글자/토큰 수 기준으로 단순 분할 (우리가 사용)
2. Sentence Chunking     → 문장 단위로 분할
3. Semantic Chunking     → 의미 단위로 분할 (가장 고급)
4. Structure Chunking    → 제목/단락 구조 기반 분할
"""

from dataclasses import dataclass
from rag.document_loader import Document


@dataclass
class Chunk:
    """
    분할된 텍스트 청크 데이터 클래스

    실무 포인트:
    chunk_index, total_chunks로 청크의 위치를 추적한다.
    검색 결과에서 앞뒤 청크를 함께 반환하는
    "Context Window 확장" 기법에 활용된다.
    """
    content: str         # 청크 텍스트
    chunk_index: int     # 전체 청크 중 몇 번째
    total_chunks: int    # 전체 청크 수
    page_num: int        # 원본 페이지 번호
    source: str          # 원본 파일명
    metadata: dict       # 추가 메타데이터


class TextChunker:
    """
    텍스트 청킹 클래스

    실무 팁:
    chunk_size = 500 토큰이 일반적인 시작점이다.
    한글은 영어보다 토큰을 많이 소모하므로
    한글 문서는 chunk_size를 조금 줄이는 게 좋다.

    최적값은 문서 특성에 따라 다르므로
    실무에서는 여러 값을 테스트해서 결정한다. (RAGAS 평가 활용)
    """

    def __init__(
        self,
        chunk_size: int = 500,      # 청크 최대 글자 수
        chunk_overlap: int = 50     # 청크 간 겹치는 글자 수
    ):
        """
        Args:
            chunk_size   : 청크 최대 크기 (글자 수 기준)
                           너무 크면 → 관련 없는 내용 포함, 비용 증가
                           너무 작면 → 맥락 부족, 검색 품질 저하
            chunk_overlap: 청크 간 겹치는 크기
                           맥락 연속성 보장을 위해 필요
                           보통 chunk_size의 10~20%
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        print(
            f"✂️  청커 초기화 완료 "
            f"(크기: {chunk_size}, 오버랩: {chunk_overlap})"
        )

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """
        Document 리스트를 Chunk 리스트로 변환

        Args:
            documents: DocumentLoader가 반환한 Document 리스트

        Returns:
            Chunk 리스트
        """

        all_chunks = []

        for doc in documents:
            chunks = self._split_text(doc)
            all_chunks.extend(chunks)

        # 전체 청크 인덱스 재계산
        total = len(all_chunks)
        for i, chunk in enumerate(all_chunks):
            chunk.chunk_index = i
            chunk.total_chunks = total

        print(f"✂️  청킹 완료: {len(documents)}페이지 → {total}개 청크\n")
        return all_chunks

    def _split_text(self, doc: Document) -> list[Chunk]:
        """
        단일 Document를 Chunk 리스트로 분할

        고정 크기 청킹 알고리즘:
        1. chunk_size 간격으로 시작점 이동
        2. 각 시작점에서 chunk_size만큼 텍스트 추출
        3. overlap만큼 다음 청크와 겹치게 설정
        """

        text = doc.content
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # 청크 텍스트 추출
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(Chunk(
                    content=chunk_text,
                    chunk_index=0,       # 나중에 재계산
                    total_chunks=0,      # 나중에 재계산
                    page_num=doc.page_num,
                    source=doc.source,
                    metadata={
                        **doc.metadata,
                        "chunk_start": start,
                        "chunk_end": end,
                    }
                ))

            # 다음 청크 시작점 = 현재 끝 - overlap
            # overlap만큼 겹쳐서 맥락 연속성 보장
            start += self.chunk_size - self.chunk_overlap

        return chunks