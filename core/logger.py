"""
구조화된 로깅 모듈

실무 포인트:
print()로 디버깅하는 건 개발할 때만 통한다.
실제 서비스에서는 로그가 유일한 디버깅 수단이야.

구조화된 로깅이란?
일반 로그:
  "에러 발생"
  → 언제? 어디서? 왜? 알 수 없음

구조화된 로그:
  {
    "timestamp": "2026-03-12T11:20:00",
    "level": "ERROR",
    "module": "rag_chain",
    "event": "document_load_failed",
    "file_path": "data/test.pdf",
    "error": "FileNotFoundError"
  }
  → 모든 컨텍스트가 한 번에!

실무에서 로그로 추적하는 것들:
- LLM 호출 시간 (느린 요청 감지)
- 토큰 사용량 (비용 추적)
- 에러 발생 위치 (디버깅)
- 사용자 패턴 (서비스 개선)
"""

import logging
import json
import time
from pathlib import Path
from datetime import datetime
from functools import wraps
from core.config import get_app_config


def setup_logger(name: str) -> logging.Logger:
    """
    모듈별 로거 생성

    Args:
        name: 로거 이름 (보통 __name__ 사용)

    Returns:
        설정된 Logger 객체
    """

    config = get_app_config()

    # 로그 디렉토리 생성
    log_dir = Path(config.log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    # 이미 핸들러가 있으면 중복 추가 방지
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, config.log_level))

    # 포맷터 설정
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 파일 핸들러
    file_handler = logging.FileHandler(
        config.log_file,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


class LLMLogger:
    """
    LLM 특화 로거

    일반 로거 + LLM 전용 메서드 추가:
    - log_llm_call: LLM 호출 기록
    - log_token_usage: 토큰 사용량 기록
    - log_rag_search: RAG 검색 기록
    """

    def __init__(self, name: str):
        self.logger = setup_logger(name)

    def log_llm_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        success: bool = True
    ) -> None:
        """LLM 호출 정보 기록"""
        data = {
            "event": "llm_call",
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "latency_ms": round(latency_ms, 2),
            "success": success
        }
        if success:
            self.logger.info(json.dumps(data, ensure_ascii=False))
        else:
            self.logger.error(json.dumps(data, ensure_ascii=False))

    def log_rag_search(
        self,
        query: str,
        num_results: int,
        top_score: float,
        latency_ms: float
    ) -> None:
        """RAG 검색 정보 기록"""
        data = {
            "event": "rag_search",
            "query_length": len(query),
            "num_results": num_results,
            "top_score": round(top_score, 4),
            "latency_ms": round(latency_ms, 2)
        }
        self.logger.info(json.dumps(data, ensure_ascii=False))

    def log_error(self, event: str, error: Exception, **kwargs) -> None:
        """에러 정보 기록"""
        data = {
            "event": event,
            "error_type": type(error).__name__,
            "error_message": str(error),
            **kwargs
        }
        self.logger.error(json.dumps(data, ensure_ascii=False))

    def info(self, msg: str, **kwargs):
        if kwargs:
            self.logger.info(f"{msg} | {json.dumps(kwargs, ensure_ascii=False)}")
        else:
            self.logger.info(msg)

    def warning(self, msg: str, **kwargs):
        if kwargs:
            self.logger.warning(f"{msg} | {json.dumps(kwargs, ensure_ascii=False)}")
        else:
            self.logger.warning(msg)

    def error(self, msg: str, **kwargs):
        if kwargs:
            self.logger.error(f"{msg} | {json.dumps(kwargs, ensure_ascii=False)}")
        else:
            self.logger.error(msg)


def log_execution_time(logger: LLMLogger = None):
    """
    함수 실행 시간 측정 데코레이터

    실무 포인트:
    느린 함수를 자동으로 감지하는 데 유용.
    API 호출, DB 쿼리 등에 붙여두면
    성능 병목을 쉽게 찾을 수 있다.

    사용법:
    @log_execution_time()
    def slow_function():
        ...
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = (time.time() - start) * 1000
                if logger:
                    logger.info(
                        f"{func.__name__} 완료",
                        latency_ms=round(elapsed, 2)
                    )
                return result
            except Exception as e:
                elapsed = (time.time() - start) * 1000
                if logger:
                    logger.log_error(
                        f"{func.__name__}_failed",
                        e,
                        latency_ms=round(elapsed, 2)
                    )
                raise
        return wrapper
    return decorator