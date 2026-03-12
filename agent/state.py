"""
Agent 상태 정의

실무 포인트:
LangGraph에서 State는 그래프 전체에서
공유되는 데이터 구조이다.

모든 노드가 State를 읽고 쓰면서
다음 노드로 전달한다.

TypedDict: 딕셔너리에 타입 힌트를 붙인 것
Annotated:  필드에 메타데이터 추가
add_messages: 메시지를 덮어쓰지 않고 누적
"""

from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Agent 전체 흐름에서 공유되는 상태

    messages:
    - 대화 히스토리 전체를 담는 리스트
    - add_messages: 새 메시지를 append (덮어쓰기 아님)
    - HumanMessage, AIMessage, ToolMessage 모두 포함
    """
    messages: Annotated[list[BaseMessage], add_messages]