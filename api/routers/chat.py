"""
채팅 라우터

실무 포인트:
라우터(Router)는 관련된 엔드포인트를 그룹으로 묶는다.
→ main.py가 복잡해지는 걸 방지
→ 기능별 파일 분리로 유지보수 용이
"""

from fastapi import APIRouter, HTTPException
from api.models import ChatRequest, ChatResponse
from langchain_app.chat import LangChainChat
from core.cache import ResponseCache
from core.logger import LLMLogger
from core.exceptions import LLMRateLimitError, LLMAPIError
import os

logger = LLMLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])

# 앱 시작 시 한 번만 초기화 (싱글톤)
_chat_instance: LangChainChat = None
_cache = ResponseCache()


def get_chat() -> LangChainChat:
    """채팅 인스턴스 싱글톤 반환"""
    global _chat_instance
    if _chat_instance is None:
        model = os.getenv("DEFAULT_MODEL", "gemini-2.0-flash-lite")
        _chat_instance = LangChainChat(model=model)
    return _chat_instance


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    일반 채팅 엔드포인트

    - 세션 ID 기반 대화 히스토리 유지
    - 응답 캐싱 적용
    - Rate Limit 에러 처리

    Request Body:
        message: 사용자 메시지
        session_id: 세션 식별자 (기본값: "default")
    """

    logger.info("채팅 요청", session_id=request.session_id)

    # 캐시 확인
    cache_key = f"{request.session_id}:{request.message}"
    cached = _cache.get(cache_key)
    if cached:
        return ChatResponse(
            response=cached,
            session_id=request.session_id,
            cached=True
        )

    try:
        chat_instance = get_chat()
        response = chat_instance.chat(
            user_input=request.message,
            session_id=request.session_id
        )

        # 캐시 저장
        _cache.set(cache_key, response)

        logger.info("채팅 응답 완료", session_id=request.session_id)
        return ChatResponse(
            response=response,
            session_id=request.session_id,
            cached=False
        )

    except Exception as e:
        # Rate Limit 에러
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            logger.log_error("rate_limit", e)
            raise HTTPException(
                status_code=429,
                detail="API 요청 한도를 초과했습니다. 잠시 후 다시 시도해주세요."
            )
        # 기타 에러
        logger.log_error("chat_error", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}")
async def clear_session(session_id: str) -> dict:
    """
    특정 세션 히스토리 초기화

    Path Parameter:
        session_id: 초기화할 세션 ID
    """
    chat_instance = get_chat()
    chat_instance.clear_memory(session_id)
    logger.info("세션 초기화", session_id=session_id)
    return {"message": f"세션 '{session_id}' 초기화 완료"}