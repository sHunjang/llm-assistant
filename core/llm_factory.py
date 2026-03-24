"""
LLM 팩토리 모듈

모델명 prefix로 어떤 LLM을 쓸지 자동 판단.
코드 변경 없이 .env 파일만 바꿔서 LLM 전환 가능.

지원 모델:
  groq/llama-3.3-70b-versatile  → Groq (기본값)
  groq/llama-3.1-8b-instant     → Groq 경량
  gemini-2.0-flash-lite          → Gemini
  gemini-2.5-flash               → Gemini 고성능
"""

import os
from langchain_core.language_models import BaseChatModel
from dotenv import load_dotenv

load_dotenv()


def create_llm(
    model: str = None,
    temperature: float = 0.7,
) -> BaseChatModel:
    """
    모델명 기반으로 적절한 LLM 인스턴스 생성

    Args:
        model: 모델명 (None이면 DEFAULT_MODEL 환경변수 사용)
        temperature: LLM 온도

    Returns:
        BaseChatModel 인스턴스
    """

    if model is None:
        model = os.getenv("DEFAULT_MODEL", "groq/llama-3.3-70b-versatile")

    # Groq 모델
    if model.startswith("groq/"):
        from langchain_groq import ChatGroq
        model_name = model.replace("groq/", "")
        return ChatGroq(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("GROQ_API_KEY")
        )

    # Gemini 모델
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
        )


# 지원 모델 목록
AVAILABLE_MODELS = {
    "groq/llama-3.3-70b-versatile": "Groq Llama 3.3 70B (기본값, 무료)",
    "groq/llama-3.1-8b-instant":    "Groq Llama 3.1 8B (빠름, 무료)",
    "gemini-2.0-flash-lite":        "Gemini 2.0 Flash Lite (무료 티어)",
    "gemini-2.5-flash":             "Gemini 2.5 Flash (고성능)",
}