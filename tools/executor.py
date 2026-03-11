"""
도구 실행 모듈

실무 포인트:
definitions.py  → "어떤 도구가 있는지" AI에게 알려주는 명세서
executor.py     → "실제로 도구를 실행"하는 엔진

이 둘을 분리한 이유:
나중에 외부 API(실제 날씨 API 등)로 교체할 때 executor.py만 수정하면 된다.
definitions.py(AI가 읽는 명세)는 건드릴 필요가 없다.
"""

import math
from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+ 내장


def get_weather(city: str, unit: str = "celsius") -> dict:
    """
    날씨 조회 함수

    실무 포인트:
    지금은 Mock 데이터를 반환하지만
    실제 서비스에서는 OpenWeatherMap, WeatherAPI 등
    실제 날씨 API를 호출한다.

    Mock을 먼저 만드는 이유:
    - API 키 없이도 전체 흐름 테스트 가능
    - 외부 API 장애 시 fallback으로 활용
    - 테스트 코드 작성이 쉬워짐
    """

    # Mock 날씨 데이터 (실제 서비스에서는 API 호출로 대체)
    mock_weather_db = {
        "서울":     {"temp_c": 18, "condition": "맑음",    "humidity": 60},
        "부산":     {"temp_c": 21, "condition": "구름 조금", "humidity": 65},
        "제주":     {"temp_c": 23, "condition": "흐림",    "humidity": 75},
        "tokyo":    {"temp_c": 20, "condition": "Cloudy",  "humidity": 70},
        "new york": {"temp_c": 15, "condition": "Rainy",   "humidity": 80},
    }

    # 도시명 정규화 (대소문자, 공백 처리)
    city_key = city.lower().strip()
    weather = mock_weather_db.get(city_key)

    if not weather:
        return {
            "success": False,
            "error": f"'{city}' 날씨 정보를 찾을 수 없습니다.",
            "available_cities": list(mock_weather_db.keys())
        }

    temp = weather["temp_c"]

    # 화씨 변환
    if unit == "fahrenheit":
        temp = round(temp * 9/5 + 32, 1)
        unit_symbol = "°F"
    else:
        unit_symbol = "°C"

    return {
        "success": True,
        "city": city,
        "temperature": f"{temp}{unit_symbol}",
        "condition": weather["condition"],
        "humidity": f"{weather['humidity']}%"
    }


def calculate(expression: str) -> dict:
    """
    수학 계산 함수

    실무 포인트:
    LLM은 큰 숫자 계산에서 오류가 발생할 수 있다.
    예: 1234567 * 9876543 → LLM이 틀린 값을 자신 있게 말하는 경우가 있음
    이 도구를 쓰면 항상 정확한 값을 보장할 수 있다.

    보안 포인트:
    eval()은 보안 위험이 있으므로
    실무에서는 허용된 연산자만 파싱하는 방식을 사용한다.
    지금은 학습 목적으로 eval()을 사용하되
    math 모듈만 허용하는 방식으로 제한한다.
    """

    try:
        # 보안: math 모듈 함수만 허용, 내장 함수 차단
        allowed_names = {
            k: v for k, v in math.__dict__.items()
            if not k.startswith("__")
        }
        allowed_names["abs"] = abs
        allowed_names["round"] = round

        result = eval(expression, {"__builtins__": {}}, allowed_names)

        return {
            "success": True,
            "expression": expression,
            "result": result
        }

    except ZeroDivisionError:
        return {
            "success": False,
            "error": "0으로 나눌 수 없습니다.",
            "expression": expression
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"계산 오류: {str(e)}",
            "expression": expression
        }


def get_current_time(timezone: str = "Asia/Seoul") -> dict:
    """
    현재 시간 조회 함수

    실무 포인트:
    글로벌 서비스에서는 시간대 처리가 매우 중요하다.
    항상 UTC 기준으로 저장하고, 표시할 때만 로컬 시간으로 변환하는 게
    실무 표준이다.
    """

    try:
        tz = ZoneInfo(timezone)
        now = datetime.now(tz)

        return {
            "success": True,
            "timezone": timezone,
            "datetime": now.strftime("%Y년 %m월 %d일 %H시 %M분 %S초"),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "weekday": ["월", "화", "수", "목", "금", "토", "일"][now.weekday()]
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"시간 조회 오류: {str(e)}",
            "timezone": timezone
        }


# ─────────────────────────────────────────
# 도구 실행 라우터
# ─────────────────────────────────────────
def execute_tool(tool_name: str, tool_args: dict) -> dict:
    """
    도구 이름을 받아서 실제 함수를 실행하는 라우터

    실무 포인트:
    이 패턴을 "디스패처(Dispatcher)" 라고 한다.
    도구가 늘어날수록 이 함수에만 추가하면 된다.
    main.py는 수정할 필요가 없다.

    Args:
        tool_name: 실행할 도구 이름
        tool_args: 도구에 전달할 인자 딕셔너리

    Returns:
        도구 실행 결과 딕셔너리
    """

    # 등록된 도구 맵
    tool_map = {
        "get_weather":      get_weather,
        "calculate":        calculate,
        "get_current_time": get_current_time,
    }

    tool_func = tool_map.get(tool_name)

    if not tool_func:
        return {
            "success": False,
            "error": f"'{tool_name}' 도구를 찾을 수 없습니다.",
            "available_tools": list(tool_map.keys())
        }

    return tool_func(**tool_args)