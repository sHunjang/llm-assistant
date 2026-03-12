"""
LangGraph Agent 그래프 정의

실무 포인트:
LangGraph의 핵심은 "상태 기반 그래프"이다.

그래프 구조:
  [START]
     ↓
  [agent_node]   ← LLM이 판단 (도구 필요? 답변 가능?)
     ↓
  조건부 엣지 판단
  ├── 도구 필요  → [tool_node] → [agent_node] (루프)
  └── 답변 완성 → [END]
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
import os

from agent.state import AgentState
from agent.tools import TOOLS

load_dotenv()


def create_agent_graph(
    model: str = None,
    system_prompt: str = None,
    temperature: float = 0.7,
):
    """
    LangGraph Agent 그래프 생성 함수

    그래프 구성:
    1. LLM에 도구 목록 바인딩
    2. agent_node: LLM 호출 노드
    3. tool_node: 도구 실행 노드
    4. 조건부 엣지: 도구 필요 여부 판단
    5. 그래프 컴파일

    Args:
        model: 사용할 Gemini 모델명
        system_prompt: 시스템 프롬프트
        temperature: LLM 온도

    Returns:
        컴파일된 LangGraph 앱
    """

    if model is None:
        model = os.getenv("DEFAULT_MODEL", "gemini-2.0-flash-lite")

    if system_prompt is None:
        system_prompt = (
            "당신은 유능한 AI 어시스턴트입니다. "
            "질문에 답하기 위해 필요한 도구를 적극적으로 활용하세요. "
            "도구 사용 후에는 결과를 바탕으로 친절하게 답변하세요. "
            "한국어로 답변하세요."
        )

    # ── 1. LLM + 도구 바인딩 ────────────────────
    # bind_tools: LLM에게 "이런 도구들을 쓸 수 있어" 알려줌
    # 2단계 Function Calling과 동일한 개념
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
    )
    llm_with_tools = llm.bind_tools(TOOLS)

    # ── 2. agent_node 정의 ───────────────────────
    # LLM을 호출하는 노드
    # State에서 messages를 읽어서 LLM에 전달
    # LLM 응답을 다시 State에 저장
    def agent_node(state: AgentState):
        """
        LLM 호출 노드

        실무 포인트:
        시스템 프롬프트를 매 호출마다 앞에 붙여줌
        → ConversationManager에서 직접 하던 것과 동일
        """
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # ── 3. tool_node 정의 ────────────────────────
    # LangGraph 내장 ToolNode 사용
    # LLM이 tool_call을 요청하면 자동으로 실행
    # 2단계 executor.py를 직접 만들었던 것과 동일한 역할
    tool_node = ToolNode(tools=TOOLS)

    # ── 4. 그래프 구성 ───────────────────────────
    graph_builder = StateGraph(AgentState)

    # 노드 추가
    graph_builder.add_node("agent", agent_node)
    graph_builder.add_node("tools", tool_node)

    # 엣지 추가
    # START → agent: 항상 agent 노드에서 시작
    graph_builder.add_edge(START, "agent")

    # agent → 조건부:
    # tools_condition이 자동으로 판단
    # → tool_call 있으면 "tools" 노드로
    # → tool_call 없으면 END로
    graph_builder.add_conditional_edges(
        "agent",
        tools_condition,    # LangGraph 내장 조건 함수
    )

    # tools → agent: 도구 실행 후 다시 agent로
    graph_builder.add_edge("tools", "agent")

    # ── 5. 그래프 컴파일 ─────────────────────────
    return graph_builder.compile()