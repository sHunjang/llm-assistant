"""
커스텀 예외 클래스 정의

실무 포인트:
Python 기본 Exception만 쓰면
어디서 어떤 종류의 에러가 났는지 파악하기 어렵다.

커스텀 예외를 만들면:
1. 에러 종류를 코드로 명확히 구분
2. 로그에서 빠르게 원인 파악
3. 에러별 다른 처리 로직 적용 가능

계층 구조:
LLMBaseError
├── LLMAPIError        → API 호출 실패
├── LLMRateLimitError  → 요청 한도 초과
├── RAGError           → RAG 파이프라인 오류
│   ├── DocumentLoadError  → 문서 로딩 실패
│   └── EmbeddingError     → 임베딩 생성 실패
└── ConfigError        → 설정 오류
"""


class LLMBaseError(Exception):
    """모든 LLM 관련 에러의 베이스 클래스"""

    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self):
        if self.details:
            return f"{self.message} | details: {self.details}"
        return self.message


class LLMAPIError(LLMBaseError):
    """
    LLM API 호출 실패

    발생 시점:
    - API 서버 장애
    - 잘못된 요청 형식
    - 인증 실패
    """
    pass


class LLMRateLimitError(LLMBaseError):
    """
    API 요청 한도 초과 (429 에러)

    발생 시점:
    - 무료 티어 일일 한도 초과
    - 분당 요청 수 초과

    실무 포인트:
    RateLimitError는 재시도 로직과 함께 처리.
    retry_after 필드로 대기 시간을 전달.
    """

    def __init__(self, message: str, retry_after: int = 60):
        super().__init__(message, {"retry_after": retry_after})
        self.retry_after = retry_after


class RAGError(LLMBaseError):
    """RAG 파이프라인 관련 에러 베이스"""
    pass


class DocumentLoadError(RAGError):
    """
    문서 로딩 실패

    발생 시점:
    - 파일이 존재하지 않음
    - 손상된 PDF
    - 지원하지 않는 파일 형식
    """
    pass


class EmbeddingError(RAGError):
    """
    임베딩 생성 실패

    발생 시점:
    - 임베딩 모델 로딩 실패
    - 빈 텍스트 입력
    """
    pass


class ConfigError(LLMBaseError):
    """
    설정 오류

    발생 시점:
    - 필수 환경변수 누락
    - 잘못된 설정값
    """
    pass