"""
LangGraph Agent 메인 실행 파일

학습 포인트:
- ReAct 패턴 동작 확인
- 도구 호출 흐름 시각화
- 2단계 Function Calling과 비교
"""

import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from agent.graph import create_agent_graph

load_dotenv()

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-2.0-flash-lite")


def print_welcome() -> None:
    print("""
╔══════════════════════════════════════════╗
║     🤖 LangGraph AI Agent                ║
║     Powered by Gemini  v5.0              ║
╚══════════════════════════════════════════╝
""")


def print_message_flow(response: dict) -> None:
    """
    Agent의 메시지 흐름을 시각화

    실무 포인트:
    Agent가 내부적으로 어떤 판단을 했는지
    투명하게 보여주는 게 디버깅에 중요하다.
    """

    messages = response.get("messages", [])

    for msg in messages:
        # AI 메시지 (도구 호출 또는 최종 답변)
        if isinstance(msg, AIMessage):
            # 도구 호출 요청인 경우
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"\n   🔧 도구 호출: {tc['name']}")
                    print(f"      인자: {tc['args']}")
            # 최종 답변인 경우
            elif msg.content:
                print(f"\n🤖 Agent: {msg.content}")

        # 도구 실행 결과
        elif isinstance(msg, ToolMessage):
            print(f"   📥 도구 결과: {msg.content}")


def run_agent_chat() -> None:
    """
    Agent 채팅 실행

    학습 포인트:
    stream_mode="values"를 사용하면
    각 노드 실행 후 중간 상태를 확인할 수 있어.
    → Agent가 어떻게 생각하는지 볼 수 있음
    """

    print(f"\n📌 모드: LangGraph ReAct Agent")
    print(f"   사용 모델: {DEFAULT_MODEL}")
    print(f"\n💡 테스트 질문 예시:")
    print(f"   - 서울 날씨 알려줘")
    print(f"   - 1234 * 5678 계산해줘")
    print(f"   - 지금 몇 시야?")
    print(f"   - LangGraph가 뭐야?")
    print(f"   - 서울 날씨 알려주고 현재 시간도 알려줘  ← 멀티 도구!")
    print(f"\n커맨드: /quit → 종료\n")

    # Agent 그래프 생성
    app = create_agent_graph(model=DEFAULT_MODEL)

    # 대화 히스토리 (세션 유지)
    conversation_history = []

    while True:
        user_input = input("\n👤 나: ").strip()
        if not user_input:
            continue
        if user_input == "/quit":
            break

        # 현재 입력을 히스토리에 추가
        conversation_history.append(HumanMessage(content=user_input))

        print("\n⚙️  Agent 실행 중...\n")

        try:
            # 그래프 실행
            # stream_mode="values": 각 노드 실행 후 전체 상태 반환
            final_response = None
            for state in app.stream(
                {"messages": conversation_history},
                stream_mode="values"
            ):
                final_response = state

            # 흐름 출력
            if final_response:
                # 새로 추가된 메시지만 출력 (입력 메시지 제외)
                new_messages = final_response["messages"][len(conversation_history):]
                print_message_flow({"messages": new_messages})

                # AI 최종 답변을 히스토리에 추가
                last_msg = final_response["messages"][-1]
                if isinstance(last_msg, AIMessage) and last_msg.content:
                    conversation_history.append(last_msg)

        except Exception as e:
            print(f"\n❌ 오류: {e}")


def run_agent_demo() -> None:
    """
    Agent 동작 데모 (자동 실행)

    학습 포인트:
    미리 정해진 질문으로 Agent 동작을 확인해봐.
    도구 호출 흐름이 어떻게 되는지 집중해서 봐.
    """

    print(f"\n📌 모드: Agent 동작 데모 (자동 실행)")
    print(f"{'='*50}\n")

    app = create_agent_graph(model=DEFAULT_MODEL)

    # 테스트 질문 목록
    test_questions = [
        "지금 몇 시야?",
        "1234 곱하기 5678은 얼마야?",
        "서울이랑 부산 날씨를 비교해줘",             # 도구 2번 호출
        "LangGraph가 뭔지 검색하고 현재 시간도 알려줘",  # 도구 2번 호출
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"[질문 {i}] {question}")
        print("-" * 40)

        try:
            final_response = None
            for state in app.stream(
                {"messages": [HumanMessage(content=question)]},
                stream_mode="values"
            ):
                final_response = state

            if final_response:
                # 입력 제외한 메시지만 출력
                new_messages = final_response["messages"][1:]
                print_message_flow({"messages": new_messages})

        except Exception as e:
            print(f"❌ 오류: {e}")

        print(f"\n{'='*50}\n")

        # Rate Limit 방지
        import time
        time.sleep(2)


def main() -> None:
    print_welcome()

    print("실행할 모드를 선택하세요:")
    print("1. Agent 채팅    (직접 대화)")
    print("2. Agent 데모    (자동 실행)")

    while True:
        choice = input("\n선택 (1/2): ").strip()
        if choice == "1":
            run_agent_chat()
            break
        elif choice == "2":
            run_agent_demo()
            break
        else:
            print("1 또는 2를 선택해줘.")


if __name__ == "__main__":
    main()