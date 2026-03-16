"""
RAG 라우터

실무 포인트:
파일 업로드는 FastAPI의 UploadFile을 사용.
PDF를 임시 파일로 저장 후 처리 → 처리 후 삭제.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from api.models import RAGQueryRequest, RAGIndexResponse, RAGQueryResponse
from langchain_app.rag_chain import LangChainRAG
from core.logger import LLMLogger
import tempfile
import os
import shutil

logger = LLMLogger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG"])

# RAG 인스턴스 싱글톤
_rag_instance: LangChainRAG = None


def get_rag() -> LangChainRAG:
    """RAG 인스턴스 싱글톤 반환"""
    global _rag_instance
    if _rag_instance is None:
        model = os.getenv("DEFAULT_MODEL", "gemini-2.0-flash-lite")
        _rag_instance = LangChainRAG(model=model)
    return _rag_instance


@router.post("/index", response_model=RAGIndexResponse)
async def index_document(
    file: UploadFile = File(..., description="인덱싱할 PDF 파일")
) -> RAGIndexResponse:
    """
    PDF 문서 인덱싱 엔드포인트

    - PDF 파일을 업로드하면 자동으로 청킹 + 임베딩 + Vector DB 저장
    - 이후 /rag/query로 질문 가능

    Request: multipart/form-data
        file: PDF 파일
    """

    # PDF 파일 검증
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="PDF 파일만 업로드 가능합니다."
        )

    logger.info("문서 인덱싱 요청", filename=file.filename)

    # 임시 파일로 저장
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # 인덱싱 실행
        rag = get_rag()
        chunks = rag.index_document(tmp_path)

        logger.info("인덱싱 완료", filename=file.filename, chunks=chunks)
        return RAGIndexResponse(
            message="문서 인덱싱 완료",
            chunks=chunks,
            filename=file.filename
        )

    except Exception as e:
        logger.log_error("index_error", e, filename=file.filename)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 임시 파일 반드시 삭제
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.post("/query", response_model=RAGQueryResponse)
async def query_document(request: RAGQueryRequest) -> RAGQueryResponse:
    """
    RAG 질문 엔드포인트

    - 인덱싱된 문서 기반으로 질문에 답변
    - 반드시 /rag/index 먼저 실행 필요

    Request Body:
        question: 질문 텍스트
        top_k: 검색 결과 수 (기본값: 3)
    """

    rag = get_rag()

    if rag.chain is None:
        raise HTTPException(
            status_code=400,
            detail="문서가 인덱싱되지 않았습니다. /rag/index를 먼저 실행하세요."
        )

    logger.info("RAG 질문", question=request.question[:30])

    try:
        answer = rag.ask(request.question)
        return RAGQueryResponse(
            answer=answer,
            question=request.question
        )

    except Exception as e:
        logger.log_error("rag_query_error", e)
        raise HTTPException(status_code=500, detail=str(e))