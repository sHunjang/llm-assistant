"""
도구(Tool) 스펙 정의 모듈

실무 포인트:
Function Calling에서 가장 중요한 건 "도구 설명"이다.
AI는 이 설명을 읽고 언제 어떤 도구를 써야 할지 판단한다.
설명이 애매하면 AI가 잘못된 도구를 선택하거나 아예 안 쓴다.

Gemini Function Calling 구조:
{
    "name": "함수명",
    "description": "AI가 읽는 설명 (매우 중요)",
    "parameters": {
        "type": "object",
        "properties": {
            "파라미터명": {
                "type": "타입",
                "description": "파라미터 설명"
            }
        },
        "required": ["필수 파라미터 목록"]
    }
}
"""

# ─────────────────────────────────────────
# 날씨 조회 도구
# ─────────────────────────────────────────
get_weather_spec = {
    "name": "get_weather",
    "description": (
        "특정 도시의 현재 날씨 정보를 조회한다. "
        "사용자가 날씨, 기온, 온도, 비, 맑음 등 "
        "날씨 관련 질문을 할 때 사용한다."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "날씨를 조회할 도시 이름 (예: 서울, 부산, Tokyo, New York)"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "온도 단위. 기본값은 celsius(섭씨)"
            }
        },
        "required": ["city"]
    }
}

# ─────────────────────────────────────────
# 계산기 도구
# ─────────────────────────────────────────
calculate_spec = {
    "name": "calculate",
    "description": (
        "수학 계산을 정확하게 수행한다. "
        "사칙연산, 제곱, 나머지 등 수학 연산이 필요할 때 사용한다. "
        "AI가 직접 계산하지 말고 반드시 이 도구를 사용해야 한다."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": (
                    "계산할 수식 문자열. "
                    "예: '1234 * 5678', '(10 + 20) / 3', '2 ** 10'"
                )
            }
        },
        "required": ["expression"]
    }
}

# ─────────────────────────────────────────
# 현재 시간 조회 도구
# ─────────────────────────────────────────
get_current_time_spec = {
    "name": "get_current_time",
    "description": (
        "현재 날짜와 시간을 조회한다. "
        "사용자가 지금 몇 시인지, 오늘 날짜가 뭔지 물어볼 때 사용한다."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "조회할 시간대 (예: Asia/Seoul, UTC, America/New_York)"
            }
        },
        "required": []
    }
}

# ─────────────────────────────────────────
# 전체 도구 목록 (LLMClient에 전달할 형식)
# ─────────────────────────────────────────
ALL_TOOLS = [
    get_weather_spec,
    calculate_spec,
    get_current_time_spec,
]