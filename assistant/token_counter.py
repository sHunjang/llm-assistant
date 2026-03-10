"""
토큰 카운터 및 비용 추적 모듈

실무 포인트:
Gemini는 현재 무료 티어이지만
토큰 사용량을 추적하는 습관은 반드시 필요하다.
나중에 유료 모델로 전환할 때 비용 예측이 가능해진다.

※ Gemini 토큰 계산 방식
   한글 1글자 ≈ 1~2 토큰
   영어 1단어 ≈ 1~2 토큰
   정확한 값은 Gemini API의 count_tokens()로 확인 가능
"""

from dataclasses import dataclass
from google import genai


# Gemini 모델별 가격 (USD per 1M tokens, 2025년 기준)
# 무료 티어는 0원이지만 추후 유료 전환 시 참고용
MODEL_PRICING = {
    "gemini-1.5-flash": {
        "input": 0.075,   # $0.075 / 1M tokens
        "output": 0.30,   # $0.30  / 1M tokens
    },
    "gemini-1.5-pro": {
        "input": 1.25,    # $1.25  / 1M tokens
        "output": 5.00,   # $5.00  / 1M tokens
    },
}

USD_TO_KRW = 1380


@dataclass
class UsageStats:
    """세션 사용량 통계"""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_requests: int = 0
    session_cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def session_cost_krw(self) -> float:
        return self.session_cost_usd * USD_TO_KRW


class TokenCounter:
    """
    토큰 카운터 및 비용 계산기

    실무 팁:
    프로덕션 환경에서는 이 데이터를 DB에 저장하고
    Grafana / 자체 대시보드로 모니터링한다.
    LangSmith, Langfuse 같은 LLM 전용 모니터링 툴도 많이 사용한다.
    """

    def __init__(self, model: str = "gemini-1.5-flash"):
        self.model = model
        self.stats = UsageStats()

    def update_usage(
        self,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        API 응답 후 사용량 업데이트 및 비용 계산

        Args:
            input_tokens : 입력 토큰 수
            output_tokens: 출력 토큰 수

        Returns:
            이번 요청의 비용 (USD)
        """
        self.stats.total_input_tokens += input_tokens
        self.stats.total_output_tokens += output_tokens
        self.stats.total_requests += 1

        # 비용 계산
        pricing = MODEL_PRICING.get(self.model, MODEL_PRICING["gemini-1.5-flash"])
        request_cost = (
            (input_tokens  / 1_000_000) * pricing["input"] +
            (output_tokens / 1_000_000) * pricing["output"]
        )
        self.stats.session_cost_usd += request_cost

        return request_cost

    def print_stats(self) -> None:
        """현재 세션 사용량 통계 출력"""
        print("\n" + "=" * 40)
        print("📊 세션 사용량 통계")
        print("=" * 40)
        print(f"총 요청 수      : {self.stats.total_requests:,} 회")
        print(f"입력 토큰       : {self.stats.total_input_tokens:,} 개")
        print(f"출력 토큰       : {self.stats.total_output_tokens:,} 개")
        print(f"총 토큰         : {self.stats.total_tokens:,} 개")
        print(f"예상 비용 (USD) : ${self.stats.session_cost_usd:.6f}")
        print(f"예상 비용 (KRW) : ₩{self.stats.session_cost_krw:.2f}")
        print("(※ 현재 무료 티어 사용 중 — 참고용 수치)")
        print("=" * 40)