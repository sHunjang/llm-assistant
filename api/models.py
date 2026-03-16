"""
API 요청/응답 스키마 정의

실무 포인트:
FastAPI는 Pydantic 모델로
요청/응답 데이터를 자동 검증한다.

장점:
1. 잘못된 요청 자동 차단 (타입 불일치, 필수값 누락)
2. 자동 API 문서 생성 (Swagger UI)
3. IDE 자동완성 지원
"""

from pydantic import BaseModel, Field
from typing import Optional


# ── 요청 스키마 ──────────────────────────────

class ChatRequest(BaseModel):
    """채팅 요청"""
    message: str = Field(..., min_length=1, description="사용자 메시지")
    session_id: str = Field(default="default", description="세션 ID")
    system_prompt: Optional[str] = Field(default=None, description="시스템 프롬프트")

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "LangChain이 뭐야?",
                "session_id": "user_001"
            }
        }
    }


class RAGQueryRequest(BaseModel):
    """RAG 질문 요청"""
    question: str = Field(..., min_length=1, description="질문")
    top_k: Optional[int] = Field(default=3, ge=1, le=10, description="검색 결과 수")

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "이 논문의 핵심 기여가 뭐야?",
                "top_k": 3
            }
        }
    }


class AgentRequest(BaseModel):
    """Agent 실행 요청"""
    message: str = Field(..., min_length=1, description="사용자 메시지")
    session_id: str = Field(default="default", description="세션 ID")

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "서울 날씨 알려주고 현재 시간도 알려줘",
                "session_id": "user_001"
            }
        }
    }


# ── 응답 스키마 ──────────────────────────────

class ChatResponse(BaseModel):
    """채팅 응답"""
    response: str = Field(..., description="AI 응답")
    session_id: str = Field(..., description="세션 ID")
    cached: bool = Field(default=False, description="캐시 히트 여부")


class RAGIndexResponse(BaseModel):
    """RAG 인덱싱 응답"""
    message: str = Field(..., description="처리 결과 메시지")
    chunks: int = Field(..., description="생성된 청크 수")
    filename: str = Field(..., description="처리된 파일명")


class RAGQueryResponse(BaseModel):
    """RAG 질문 응답"""
    answer: str = Field(..., description="문서 기반 답변")
    question: str = Field(..., description="원본 질문")


class AgentResponse(BaseModel):
    """Agent 실행 응답"""
    response: str = Field(..., description="Agent 최종 답변")
    session_id: str = Field(..., description="세션 ID")
    tools_used: list[str] = Field(
        default_factory=list,
        description="사용된 도구 목록"
    )


class HealthResponse(BaseModel):
    """서버 상태 응답"""
    status: str = Field(..., description="서버 상태")
    version: str = Field(..., description="API 버전")
    model: str = Field(..., description="사용 중인 LLM 모델")