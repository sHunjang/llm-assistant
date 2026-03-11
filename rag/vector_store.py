"""
Vector DB 관리 모듈 (ChromaDB)

실무 포인트:
Vector DB는 RAG 시스템의 "지식 창고"다.
한 번 구축해두면 여러 질문에 재사용할 수 있다.

ChromaDB 선택 이유:
- 로컬 실행 가능 (API 키 불필요)
- 설치가 간단 (pip install chromadb)
- 영구 저장 지원 (파일 기반)
- 학습/프로토타입에 최적

실무에서 ChromaDB → Pinecone/Qdrant 전환 시
vector_store.py만 교체하면 됨 (인터페이스 통일)
"""

import chromadb
from chromadb.config import Settings


class VectorStore:
    """
    ChromaDB 기반 Vector Store 클래스

    두 가지 모드:
    1. 영구 저장 모드 (persist_directory 지정)
       → 프로그램 종료 후에도 데이터 유지
       → 실무에서 사용

    2. 인메모리 모드 (persist_directory=None)
       → 프로그램 종료 시 데이터 사라짐
       → 테스트/개발용
    """

    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: str = "./chroma_db"
    ):
        """
        Args:
            collection_name  : ChromaDB 컬렉션 이름
                               (RDB의 테이블명과 유사)
            persist_directory: DB 파일 저장 경로
                               None이면 인메모리 모드
        """

        self.collection_name = collection_name

        # ChromaDB 클라이언트 초기화
        if persist_directory:
            # 영구 저장 모드
            self.client = chromadb.PersistentClient(
                path=persist_directory
            )
            print(f"💾 ChromaDB 초기화 (영구 저장: {persist_directory})")
        else:
            # 인메모리 모드
            self.client = chromadb.Client()
            print("💾 ChromaDB 초기화 (인메모리 모드)")

        # 컬렉션 가져오기 또는 생성
        # get_or_create: 있으면 가져오고 없으면 새로 만들기
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # 코사인 유사도 사용
        )

        print(
            f"✅ 컬렉션 준비 완료: '{collection_name}' "
            f"(현재 {self.collection.count()}개 문서)\n"
        )

    def add_documents(self, embedded_docs: list[dict]) -> None:
        """
        임베딩된 문서를 Vector DB에 저장

        Args:
            embedded_docs: Embedder.embed_chunks()가 반환한 딕셔너리 리스트
        """

        if not embedded_docs:
            print("⚠️  저장할 문서가 없습니다.")
            return

        # ChromaDB add() 형식으로 변환
        ids        = [doc["id"]        for doc in embedded_docs]
        contents   = [doc["content"]   for doc in embedded_docs]
        embeddings = [doc["embedding"] for doc in embedded_docs]
        metadatas  = [doc["metadata"]  for doc in embedded_docs]

        # 중복 ID 처리: upsert = 있으면 업데이트, 없으면 삽입
        self.collection.upsert(
            ids=ids,
            documents=contents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(f"💾 {len(embedded_docs)}개 문서 저장 완료\n")

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3
    ) -> list[dict]:
        """
        쿼리 임베딩과 유사한 문서 검색

        실무 포인트:
        top_k는 검색 결과 수다.
        너무 작으면 → 관련 문서 누락
        너무 크면  → 관련 없는 문서 포함, 토큰 낭비

        일반적으로 3~5개가 적절하다.

        Args:
            query_embedding: 질문의 임베딩 벡터
            top_k          : 반환할 최대 결과 수

        Returns:
            유사도 순으로 정렬된 검색 결과 리스트
        """

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        # 결과 파싱 및 정리
        search_results = []
        for i in range(len(results["ids"][0])):
            # distance → similarity 변환
            # ChromaDB 코사인 거리: 0(동일) ~ 2(반대)
            # similarity로 변환: 1 - distance/2
            distance   = results["distances"][0][i]
            similarity = round(1 - distance / 2, 4)

            search_results.append({
                "content":    results["documents"][0][i],
                "metadata":   results["metadatas"][0][i],
                "similarity": similarity
            })

        return search_results

    def get_document_count(self) -> int:
        """저장된 문서 수 반환"""
        return self.collection.count()

    def clear(self) -> None:
        """컬렉션의 모든 문서 삭제"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print("🗑️  Vector DB 초기화 완료")