"""
Agent 도구 정의

실무 포인트:
2단계에서 Function Calling 도구를 직접 정의했던 것과 비교.

2단계 방식:
  definitions.py → 딕셔너리로 스펙 정의
  executor.py    → 별도 클래스로 실행

LangGraph 방식:
  @tool 데코레이터 하나로 정의 + 실행 통합
  → LLM이 자동으로 함수 시그니처를 읽어서 호출
"""

from langchain_core.tools import tool
from datetime import datetime
import random


@tool
def get_weather(city: str) -> str:
    """
    특정 도시의 현재 날씨를 조회한다.

    Args:
        city: 날씨를 조회할 도시 이름 (예: 서울, 부산, 제주)

    Returns:
        날씨 정보 문자열
    """
    # 실무에서는 실제 날씨 API 호출
    # 여기서는 Mock 데이터로 대체
    weather_data = {
        "서울": {"temp": 18, "condition": "맑음", "humidity": 45},
        "부산": {"temp": 22, "condition": "구름 조금", "humidity": 60},
        "제주": {"temp": 20, "condition": "흐림", "humidity": 75},
        "대구": {"temp": 25, "condition": "맑음", "humidity": 40},
        "인천": {"temp": 17, "condition": "안개", "humidity": 80},
    }

    if city in weather_data:
        data = weather_data[city]
        return (
            f"{city} 현재 날씨: {data['condition']}, "
            f"기온 {data['temp']}°C, "
            f"습도 {data['humidity']}%"
        )
    return f"{city}의 날씨 정보를 찾을 수 없습니다."


@tool
def calculate(expression: str) -> str:
    """
    수학 계산식을 계산한다.

    Args:
        expression: 계산할 수식 문자열
                   (예: "2 + 3", "10 * 5", "100 / 4")

    Returns:
        계산 결과 문자열
    """
    try:
        # 안전한 계산을 위해 eval 대신 제한된 연산만 허용
        allowed = set('0123456789+-*/()., ')
        if not all(c in allowed for c in expression):
            return "허용되지 않는 문자가 포함되어 있습니다."

        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"계산 오류: {str(e)}"


@tool
def get_current_time(timezone: str = "KST") -> str:
    """
    현재 시간을 조회한다.

    Args:
        timezone: 시간대 (기본값: KST)

    Returns:
        현재 시간 문자열
    """
    now = datetime.now()
    return (
        f"현재 시간 ({timezone}): "
        f"{now.strftime('%Y년 %m월 %d일 %H시 %M분 %S초')}"
    )


@tool
def search_knowledge(query: str) -> str:
    """
    일반 지식을 검색한다.

    Args:
        query: 검색할 키워드나 질문

    Returns:
        관련 지식 문자열
    """
    # 실무에서는 실제 검색 API (Google, Tavily 등) 호출
    # 여기서는 Mock 데이터로 대체
    knowledge_base = {
        "LangGraph": (
            "LangGraph는 LangChain 팀이 만든 상태 기반 AI Agent 프레임워크입니다. "
            "그래프 구조로 복잡한 Agent 워크플로우를 구현할 수 있으며, "
            "조건부 분기, 루프, 병렬 처리를 지원합니다."
        ),
        "RAG": (
            "RAG(Retrieval-Augmented Generation)는 외부 문서를 검색해서 "
            "LLM의 답변 품질을 높이는 기술입니다. "
            "환각(Hallucination)을 줄이고 최신 정보를 활용할 수 있습니다."
        ),
        "LangChain": (
            "LangChain은 LLM 기반 애플리케이션 개발 프레임워크입니다. "
            "프롬프트 관리, 체인 구성, 메모리 관리 등을 제공하며 "
            "LCEL(LangChain Expression Language)로 파이프라인을 구성합니다."
        ),
    }

    # 키워드 매칭
    for key, value in knowledge_base.items():
        if key.lower() in query.lower():
            return value

    return f"'{query}'에 대한 검색 결과를 찾을 수 없습니다."


# 도구 목록 (graph.py에서 import해서 사용)
TOOLS = [get_weather, calculate, get_current_time, search_knowledge]