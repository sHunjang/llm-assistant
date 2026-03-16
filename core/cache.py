"""
응답 캐싱 모듈

실무 포인트:
LLM API는 같은 질문을 두 번 물어도
매번 API를 호출하고 비용이 발생한다.

캐싱으로 해결:
1. 첫 번째 질문 → API 호출 → 결과 저장
2. 같은 질문 재요청 → 캐시에서 즉시 반환
   → API 호출 없음 = 비용 0 + 속도 향상

캐싱이 유용한 경우:
- FAQ 챗봇: 자주 묻는 질문이 반복됨
- RAG 시스템: 같은 문서에 같은 질문 반복
- 개발/테스트: 같은 프롬프트 반복 실행

캐싱하면 안 되는 경우:
- 현재 시간, 날씨 등 실시간 정보
- 사용자별 맞춤 응답
- 매번 달라야 하는 창의적 응답
"""

import hashlib
import time
import json
from collections import OrderedDict
from core.config import get_app_config
from core.logger import LLMLogger

logger = LLMLogger(__name__)


class LRUCache:
    """
    LRU(Least Recently Used) 캐시 구현

    LRU 알고리즘:
    캐시가 꽉 찼을 때 가장 오래전에 사용된 항목을 제거.
    자주 쓰는 건 남기고, 안 쓰는 건 버리는 전략.

    실무 포인트:
    Python의 OrderedDict를 활용하면
    LRU 캐시를 간단하게 구현할 수 있어.
    """

    def __init__(self, max_size: int = 100, ttl: int = 3600):
        """
        Args:
            max_size: 최대 캐시 항목 수
            ttl: 캐시 유효 시간 (초)
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict = OrderedDict()
        self.hits = 0       # 캐시 히트 수
        self.misses = 0     # 캐시 미스 수

    def _make_key(self, text: str) -> str:
        """
        캐시 키 생성

        MD5 해시를 사용해서
        긴 텍스트를 짧은 고정 길이 키로 변환.
        """
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, key: str) -> str | None:
        """
        캐시에서 값 조회

        Returns:
            캐시된 값 (없거나 만료되면 None)
        """
        hashed_key = self._make_key(key)

        if hashed_key not in self.cache:
            self.misses += 1
            return None

        value, timestamp = self.cache[hashed_key]

        # TTL 만료 확인
        if time.time() - timestamp > self.ttl:
            del self.cache[hashed_key]
            self.misses += 1
            return None

        # LRU: 최근 사용 항목을 맨 뒤로 이동
        self.cache.move_to_end(hashed_key)
        self.hits += 1
        return value

    def set(self, key: str, value: str) -> None:
        """캐시에 값 저장"""
        hashed_key = self._make_key(key)

        # 이미 있으면 업데이트
        if hashed_key in self.cache:
            self.cache.move_to_end(hashed_key)

        self.cache[hashed_key] = (value, time.time())

        # 최대 크기 초과 시 가장 오래된 항목 제거
        if len(self.cache) > self.max_size:
            oldest = next(iter(self.cache))
            del self.cache[oldest]
            logger.info("캐시 항목 제거 (LRU)", cache_size=len(self.cache))

    def clear(self) -> None:
        """캐시 초기화"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        logger.info("캐시 초기화 완료")

    def get_stats(self) -> dict:
        """캐시 통계 반환"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "ttl_seconds": self.ttl
        }


class ResponseCache:
    """
    LLM 응답 캐싱 클래스

    LRUCache를 LLM 응답에 특화시킨 래퍼.
    설정 파일 기반으로 캐시 활성화 여부 제어 가능.
    """

    def __init__(self):
        config = get_app_config()
        self.enabled = config.cache_enabled
        self.cache = LRUCache(
            max_size=config.cache_max_size,
            ttl=config.cache_ttl
        )
        status = "활성화" if self.enabled else "비활성화"
        logger.info(f"응답 캐시 초기화 완료 ({status})")

    def get(self, prompt: str) -> str | None:
        """프롬프트로 캐시된 응답 조회"""
        if not self.enabled:
            return None
        result = self.cache.get(prompt)
        if result:
            logger.info("캐시 히트", prompt_length=len(prompt))
        return result

    def set(self, prompt: str, response: str) -> None:
        """프롬프트-응답 쌍을 캐시에 저장"""
        if not self.enabled:
            return
        self.cache.set(prompt, response)

    def get_stats(self) -> dict:
        """캐시 통계 반환"""
        return self.cache.get_stats()

    def clear(self) -> None:
        """캐시 초기화"""
        self.cache.clear()