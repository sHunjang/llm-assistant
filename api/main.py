"""
FastAPI 애플리케이션 진입점

실무 포인트:
lifespan: 서버 시작/종료 시 실행할 코드 정의
  → 시작 시: DB 연결, 모델 로딩 등 초기화
  → 종료 시: 연결 해제, 리소스 정리
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from api.routers import chat, rag, agent
from api.models import HealthResponse
from core.config import get_llm_config, get_app_config
from core.logger import LLMLogger

load_dotenv()
logger = LLMLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작/종료 시 실행"""

    # ── 시작 시 ─────────────────────────────
    logger.info("서버 시작 중...")
    config = get_llm_config()
    app_config = get_app_config()
    logger.info(
        "설정 로딩 완료",
        model=config.default_model,
        environment=app_config.environment
    )

    yield  # 서버 실행 중

    # ── 종료 시 ─────────────────────────────
    logger.info("서버 종료 중...")


# FastAPI 앱 생성
app = FastAPI(
    title="LLM Assistant API",
    description="""
## LLM Assistant API

1~6단계에서 구현한 LLM 기능들을 REST API로 제공합니다.

### 주요 기능
- **Chat**: 세션 기반 멀티턴 대화
- **RAG**: PDF 문서 기반 질의응답
- **Agent**: 도구를 활용한 자율 실행 Agent
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
# 실무 포인트:
# 프로덕션에서는 allow_origins에 실제 도메인만 허용
# 개발 중에는 "*"로 모든 도메인 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(chat.router)
app.include_router(rag.router)
app.include_router(agent.router)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """
    서버 상태 확인 엔드포인트

    실무 포인트:
    배포 환경에서 서버가 살아있는지 주기적으로 확인할 때 사용.
    로드밸런서, Kubernetes 등이 이 엔드포인트로 헬스체크를 함.
    """
    config = get_llm_config()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model=config.default_model
    )


@app.get("/", tags=["System"])
async def root() -> dict:
    """API 루트"""
    return {
        "message": "LLM Assistant API",
        "docs": "/docs",
        "health": "/health"
    }