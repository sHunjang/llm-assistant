"""
RAG 검색 파이프라인 모듈

실무 포인트:
retriever.py는 RAG의 모든 컴포넌트를 조립하는 "사령탑"이다.

document_loader → chunker → embedder → vector_store
이 파이프라인을 하나의 인터페이스로 통합한다.
"""

from rag.document_loader import DocumentLoader
from rag.chunker import TextChunker
from rag.embedder import Embedder
from rag.vector_store import VectorStore


class RAGRetriever:
    """
    RAG 검색 파이프라인 통합 클래스

    두 가지 주요 기능:
    1. index_document()  → 문서를 처리해서 Vector DB에 저장
    2. retrieve()        → 질문과 관련된 문서 청크 검색
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 3,
        persist_directory: str = "./chroma_db"
    ):
        """
        Args:
            chunk_size        : 청크 크기
            chunk_overlap     : 청크 오버랩
            top_k             : 검색 결과 수
            persist_directory : Vector DB 저장 경로
        """
        self.top_k = top_k

        # 컴포넌트 초기화
        self.loader       = DocumentLoader()
        self.chunker      = TextChunker(chunk_size, chunk_overlap)
        self.embedder     = Embedder()
        self.vector_store = VectorStore(
            persist_directory=persist_directory
        )

        print("✅ RAG Retriever 초기화 완료\n")

    def index_document(self, file_path: str) -> int:
        """
        문서 처리 파이프라인 실행
        PDF → 파싱 → 청킹 → 임베딩 → Vector DB 저장

        Args:
            file_path: 처리할 PDF 파일 경로

        Returns:
            저장된 청크 수
        """

        print(f"{'='*50}")
        print(f"📥 문서 인덱싱 시작: {file_path}")
        print(f"{'='*50}\n")

        # Step 1: PDF 로딩
        documents = self.loader.load_pdf(file_path)

        # Step 2: 청킹
        chunks = self.chunker.chunk_documents(documents)

        # Step 3: 임베딩
        embedded_docs = self.embedder.embed_chunks(chunks)

        # Step 4: Vector DB 저장
        self.vector_store.add_documents(embedded_docs)

        total = self.vector_store.get_document_count()
        print(f"✅ 인덱싱 완료! Vector DB 총 {total}개 청크 저장\n")

        return len(embedded_docs)

    def retrieve(self, query: str) -> list[dict]:
        """
        질문과 관련된 문서 청크 검색

        Args:
            query: 사용자 질문

        Returns:
            관련 문서 청크 리스트 (유사도 순 정렬)
        """

        # 질문을 임베딩 벡터로 변환
        query_embedding = self.embedder.embed_text(query)

        # Vector DB에서 유사 문서 검색
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.top_k
        )

        return results

    def format_context(self, search_results: list[dict]) -> str:
        """
        검색 결과를 LLM 프롬프트에 삽입할 컨텍스트 문자열로 변환

        실무 포인트:
        검색 결과를 그냥 붙여넣지 않고
        출처 정보와 함께 구조화된 형식으로 전달해야
        LLM이 정확한 답변과 출처를 함께 제공할 수 있다.

        Args:
            search_results: retrieve()가 반환한 검색 결과

        Returns:
            프롬프트에 삽입할 컨텍스트 문자열
        """

        if not search_results:
            return "관련 문서를 찾을 수 없습니다."

        context_parts = []
        for i, result in enumerate(search_results, start=1):
            meta       = result["metadata"]
            source     = meta.get("source", "알 수 없음")
            page_num   = meta.get("page_num", "?")
            similarity = result["similarity"]

            context_parts.append(
                f"[문서 {i}] 출처: {source} {page_num}페이지 "
                f"(관련도: {similarity:.1%})\n"
                f"{result['content']}"
            )

        return "\n\n".join(context_parts)

    def get_stats(self) -> dict:
        """RAG 시스템 현황 반환"""
        return {
            "total_chunks": self.vector_store.get_document_count(),
            "top_k": self.top_k,
            "embedding_model": self.embedder.model_name,
        }