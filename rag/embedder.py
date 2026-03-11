"""
임베딩 생성 모듈

실무 포인트:
임베딩 모델 선택이 RAG 검색 품질의 핵심이다.

임베딩 모델 선택 기준:
1. 언어 지원     → 한글 지원 여부 확인 필수
2. 벡터 차원수   → 높을수록 정확하지만 느리고 비쌈
3. 속도          → 실시간 서비스면 빠른 모델 필요
4. 비용          → API 기반 vs 로컬 모델

우리가 사용할 모델:
jhgan/ko-sroberta-multitask
→ 한국어 특화 임베딩 모델
→ 로컬 실행 (API 비용 없음)
→ 학습용으로 충분한 성능

실무에서 많이 쓰는 임베딩 모델:
- OpenAI text-embedding-3-small  → 성능 좋음, API 비용 발생
- Google text-embedding-004      → Gemini 생태계
- BGE-M3                         → 오픈소스 최강자
- ko-sroberta-multitask          → 한국어 특화 ✅ (우리 사용)
"""

from sentence_transformers import SentenceTransformer
from rag.chunker import Chunk
import numpy as np


class Embedder:
    """
    텍스트 임베딩 생성 클래스

    실무 포인트:
    임베딩 모델은 무겁기 때문에
    매번 새로 로딩하지 않고 한 번만 로딩해서 재사용한다.
    (싱글톤 패턴)
    """

    def __init__(
        self,
        model_name: str = "jhgan/ko-sroberta-multitask"
    ):
        """
        Args:
            model_name: 사용할 임베딩 모델명
                        최초 실행 시 자동으로 다운로드됨 (약 300MB)
        """
        print(f"🧠 임베딩 모델 로딩 중: {model_name}")
        print("   (최초 실행 시 모델 다운로드로 시간이 걸릴 수 있습니다)")

        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        print(f"✅ 임베딩 모델 로딩 완료 (벡터 차원: {self.embedding_dim})\n")

    def embed_text(self, text: str) -> list[float]:
        """
        단일 텍스트를 임베딩 벡터로 변환

        사용처:
        - 사용자 질문을 벡터로 변환할 때 (검색 시)

        Args:
            text: 임베딩할 텍스트

        Returns:
            임베딩 벡터 (float 리스트)
        """
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_chunks(self, chunks: list[Chunk]) -> list[dict]:
        """
        청크 리스트를 임베딩하여 저장 가능한 형태로 변환

        실무 포인트:
        청크가 많을 때는 배치(batch) 처리로 한 번에 임베딩하면
        개별 처리보다 훨씬 빠르다.

        Args:
            chunks: Chunk 리스트

        Returns:
            임베딩이 포함된 딕셔너리 리스트
            (ChromaDB에 바로 저장 가능한 형태)
        """
        print(f"🧠 {len(chunks)}개 청크 임베딩 중...")

        # 텍스트만 추출해서 배치 임베딩 (한 번에 처리 → 빠름)
        texts = [chunk.content for chunk in chunks]
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,     # 진행률 표시
            batch_size=32               # 메모리 효율적 처리
        )

        # ChromaDB 저장 형식으로 변환
        result = []
        for chunk, embedding in zip(chunks, embeddings):
            result.append({
                "id": f"{chunk.source}_p{chunk.page_num}_c{chunk.chunk_index}",
                "content": chunk.content,
                "embedding": embedding.tolist(),
                "metadata": {
                    "source": chunk.source,
                    "page_num": chunk.page_num,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    **chunk.metadata
                }
            })

        print(f"✅ 임베딩 완료: {len(result)}개 벡터 생성\n")
        return result