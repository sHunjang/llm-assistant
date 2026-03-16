"""
설정 관리 모듈

실무 포인트:
지금까지 코드 곳곳에 흩어져 있던 설정들
(모델명, temperature, chunk_size 등)을
한 곳에서 관리하는 것이 핵심이야.

Pydantic Settings 장점:
1. 환경변수 자동 로딩 (.env 파일 포함)
2. 타입 검증 자동 처리
3. 설정 누락 시 즉시 에러 (Fail Fast)
4. IDE 자동완성 지원

환경별 설정 분리:
development  → 디버그 로깅, lite 모델
production   → 에러 로깅만, 고성능 모델
testing      → Mock 설정
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal
from functools import lru_cache


class LLMConfig(BaseSettings):
    """
    LLM 관련 설정

    실무 포인트:
    Field(default=...) 로 기본값 설정
    Field(...) 로 필수값 지정 (없으면 에러)
    """

    # API 설정
    gemini_api_key: str = Field(..., description="Gemini API 키 (필수)")

    # 모델 설정
    default_model: str = Field(
        default="gemini-2.0-flash-lite",
        description="기본 LLM 모델"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,     # 최솟값
        le=2.0,     # 최댓값
        description="LLM 온도 (0.0~2.0)"
    )
    max_tokens: int = Field(
        default=1000,
        description="최대 출력 토큰 수"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"    # 정의되지 않은 환경변수 무시


class RAGConfig(BaseSettings):
    """RAG 파이프라인 설정"""

    # 청킹 설정
    chunk_size: int = Field(default=500)
    chunk_overlap: int = Field(default=50)

    # 검색 설정
    top_k: int = Field(default=3, ge=1, le=10)
    similarity_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="최소 유사도 임계값 (이하 결과 제외)"
    )

    # 임베딩 설정
    embedding_model: str = Field(
        default="jhgan/ko-sroberta-multitask"
    )

    # Vector DB 설정
    chroma_persist_dir: str = Field(default="./chroma_db")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


class AppConfig(BaseSettings):
    """애플리케이션 전체 설정"""

    # 환경 설정
    environment: Literal["development", "production", "testing"] = Field(
        default="development"
    )

    # 로깅 설정
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO"
    )
    log_file: str = Field(
        default="logs/app.log",
        description="로그 파일 경로"
    )

    # 캐시 설정
    cache_enabled: bool = Field(default=True)
    cache_ttl: int = Field(
        default=3600,
        description="캐시 유효 시간 (초)"
    )
    cache_max_size: int = Field(
        default=100,
        description="최대 캐시 항목 수"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# ── 싱글톤 패턴 ──────────────────────────────
# lru_cache: 한 번 생성된 설정 객체를 재사용
# → 매번 .env를 다시 읽지 않아도 됨
@lru_cache(maxsize=1)
def get_llm_config() -> LLMConfig:
    return LLMConfig()


@lru_cache(maxsize=1)
def get_rag_config() -> RAGConfig:
    return RAGConfig()


@lru_cache(maxsize=1)
def get_app_config() -> AppConfig:
    return AppConfig()