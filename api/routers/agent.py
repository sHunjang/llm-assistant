"""
Agent 라우터
"""

from fastapi import APIRouter, HTTPException
from api.models import AgentRequest, AgentResponse
from agent.graph import create_agent_graph
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from core.logger import LLMLogger
import os

logger = LLMLogger(__name__)

router = APIRouter(prefix="/agent", tags=["Agent"])

# Agent 그래프 싱글톤
_agent_app = None

# 세션별 대화 히스토리
_session_histories: dict = {}


def get_agent():
    """Agent 앱 싱글톤 반환"""
    global _agent_app
    if _agent_app is None:
        model = os.getenv("DEFAULT_MODEL", "gemini-2.0-flash-lite")
        _agent_app = create_agent_graph(model=model)
    return _agent_app


@router.post("", response_model=AgentResponse)
async def run_agent(request: AgentRequest) -> AgentResponse:
    """
    Agent 실행 엔드포인트

    - ReAct 패턴으로 도구를 자율적으로 선택해서 실행
    - 세션 ID 기반 대화 히스토리 유지
    - 사용된 도구 목록 반환

    Request Body:
        message: 사용자 메시지
        session_id: 세션 식별자
    """

    logger.info("Agent 요청", session_id=request.session_id)

    # 세션 히스토리 로드
    if request.session_id not in _session_histories:
        _session_histories[request.session_id] = []

    history = _session_histories[request.session_id]
    history.append(HumanMessage(content=request.message))

    try:
        app = get_agent()
        final_response = None

        for state in app.stream(
            {"messages": history},
            stream_mode="values"
        ):
            final_response = state

        if not final_response:
            raise HTTPException(status_code=500, detail="Agent 응답 없음")

        # 새 메시지에서 결과 추출
        new_messages = final_response["messages"][len(history):]
        final_answer = ""
        tools_used = []

        for msg in new_messages:
            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        tools_used.append(tc["name"])
                elif msg.content:
                    final_answer = msg.content
            elif isinstance(msg, ToolMessage):
                pass  # 도구 결과는 tools_used로 표현

        # 히스토리 업데이트
        last_msg = final_response["messages"][-1]
        if isinstance(last_msg, AIMessage) and last_msg.content:
            history.append(last_msg)

        logger.info(
            "Agent 응답 완료",
            tools_used=tools_used,
            session_id=request.session_id
        )

        return AgentResponse(
            response=final_answer,
            session_id=request.session_id,
            tools_used=tools_used
        )

    except Exception as e:
        logger.log_error("agent_error", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}")
async def clear_agent_session(session_id: str) -> dict:
    """Agent 세션 히스토리 초기화"""
    if session_id in _session_histories:
        del _session_histories[session_id]
    return {"message": f"Agent 세션 '{session_id}' 초기화 완료"}