from langchain_core.tools import tool
from datetime import datetime
import os
import requests


@tool
def get_weather(city: str) -> str:
    """
    Get current weather for a specific city.

    Args:
        city: City name to get weather for (e.g., Seoul, Busan, Jeju)

    Returns:
        Weather information string
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        weather_data = {
            "서울": {"temp": 18, "condition": "맑음", "humidity": 45},
            "부산": {"temp": 22, "condition": "구름 조금", "humidity": 60},
            "제주": {"temp": 20, "condition": "흐림", "humidity": 75},
            "대구": {"temp": 25, "condition": "맑음", "humidity": 40},
            "창원": {"temp": 20, "condition": "맑음", "humidity": 55},
        }
        if city in weather_data:
            data = weather_data[city]
            return f"{city} 현재 날씨: {data['condition']}, 기온 {data['temp']}°C, 습도 {data['humidity']}%"
        return f"{city}의 날씨 정보를 찾을 수 없습니다."

    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": api_key, "units": "metric", "lang": "kr"}
        response = requests.get(url, params=params, timeout=5)
        data = response.json()

        if response.status_code == 200:
            temp = data["main"]["temp"]
            feels_like = data["main"]["feels_like"]
            humidity = data["main"]["humidity"]
            condition = data["weather"][0]["description"]
            wind = data["wind"]["speed"]
            return f"{city} 현재 날씨: {condition}, 기온 {temp:.1f}°C (체감 {feels_like:.1f}°C), 습도 {humidity}%, 풍속 {wind}m/s"
        else:
            return f"날씨 정보를 가져올 수 없습니다: {data.get('message', '알 수 없는 오류')}"
    except Exception as e:
        return f"날씨 API 오류: {str(e)}"


@tool
def calculate(expression: str) -> str:
    """
    Calculate a mathematical expression.

    Args:
        expression: Math expression to calculate (e.g., "2 + 3", "10 * 5")

    Returns:
        Calculation result string
    """
    try:
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
    Get the current time.

    Args:
        timezone: Timezone name (default: KST)

    Returns:
        Current time string
    """
    now = datetime.now()
    return f"현재 시간 ({timezone}): {now.strftime('%Y년 %m월 %d일 %H시 %M분 %S초')}"


@tool
def search_knowledge(query: str) -> str:
    """
    Search for information about AI, technology, and general knowledge topics.
    Use this tool when asked about LangGraph, LangChain, RAG, LLM, or any topic requiring factual information.

    Args:
        query: The search keyword or question to look up (e.g., "LangGraph", "RAG", "LangChain")

    Returns:
        Search results as a string
    """
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        knowledge_base = {
            "LangGraph": "LangGraph는 LangChain 팀이 만든 상태 기반 AI Agent 프레임워크입니다. 그래프 구조로 복잡한 Agent 워크플로우를 구현할 수 있으며, 조건부 분기, 루프, 병렬 처리를 지원합니다.",
            "RAG": "RAG(Retrieval-Augmented Generation)는 외부 문서를 검색해서 LLM의 답변 품질을 높이는 기술입니다.",
            "LangChain": "LangChain은 LLM 기반 애플리케이션 개발 프레임워크입니다.",
        }
        for key, value in knowledge_base.items():
            if key.lower() in query.lower():
                return value
        return f"'{query}'에 대한 검색 결과를 찾을 수 없습니다."

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, max_results=3, search_depth="basic")
        results = response.get("results", [])
        if not results:
            return f"'{query}'에 대한 검색 결과가 없습니다."
        output = []
        for r in results[:3]:
            title = r.get("title", "")
            content = r.get("content", "")[:200]
            output.append(f"• {title}: {content}")
        return "\n".join(output)
    except Exception as e:
        return f"검색 오류: {str(e)}"


TOOLS = [get_weather, calculate, get_current_time, search_knowledge]
