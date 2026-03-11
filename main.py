"""
LLM 어시스턴트 메인 실행 파일

실행 방법:
    python main.py

사용 가능한 커맨드:
    /help   - 도움말 출력
    /clear  - 대화 히스토리 초기화
    /stats  - 토큰 사용량 및 비용 통계
    /save   - 현재 대화를 파일로 저장
    /quit   - 프로그램 종료
    
2단계 변경사항:
- 도구 실행 루프 추가
- Function Calling 흐름 구현
"""

import json
from assistant.client import LLMClient
from assistant.conversation import ConversationManager
from assistant.token_counter import TokenCounter
from prompts.system_prompts import LLM_MENTOR
from tools.definitions import ALL_TOOLS
from tools.executor import execute_tool


def print_welcome() -> None:
    """시작 화면 출력"""
    print("""
╔══════════════════════════════════════════╗
║     🤖 LLM 엔지니어링 AI 멘토              ║
║   Powered by Gemini 2.5 Flash      v2.0  ║
╚══════════════════════════════════════════╝

🛠️  사용 가능한 도구:
   날씨 조회    → "서울 날씨 알려줘"
   계산기       → "1234 * 5678 계산해줘"
   현재 시간    → "지금 몇 시야?"

💡 커맨드:
   /help   → 도움말
   /clear  → 대화 초기화
   /stats  → 토큰/비용 통계
   /save   → 대화 저장
   /quit   → 종료
""")


def handle_command(
    command: str,
    conversation: ConversationManager,
    counter: TokenCounter
) -> bool:
    """
    슬래시 커맨드 처리

    Returns:
        True  → 계속 실행
        False → 프로그램 종료
    """
    cmd = command.lower().strip()

    if cmd == "/help":
        print("""
📚 사용 가능한 커맨드:
   /help   → 이 도움말 출력
   /clear  → 대화 히스토리 초기화
   /stats  → 토큰 사용량 및 비용 통계
   /save   → 현재 대화를 파일로 저장
   /quit   → 프로그램 종료

🛠️  사용 가능한 도구:
   날씨 조회    → "서울 날씨 알려줘"
   계산기       → "1234 * 5678 계산해줘"
   현재 시간    → "지금 몇 시야?"
""")

    elif cmd == "/clear":
        conversation.clear()

    elif cmd == "/stats":
        counter.print_stats()
        print(f"💬 현재 대화 수: {conversation.get_history_count()}개")

    elif cmd == "/save":
        conversation.save_to_file()

    elif cmd in ["/quit", "/exit", "/q"]:
        print("\n📊 최종 사용량:")
        counter.print_stats()
        print("\n👋 수고하셨습니다!")
        return False

    else:
        print(f"❓ 알 수 없는 커맨드: {cmd}  (도움말: /help)")

    return True

def run_function_calling_loop(
    client: LLMClient,
    messages: list[dict],
    system_prompt: str,
    counter: TokenCounter,
) -> str:
    """
    Function Calling 실행 루프

    실무 포인트:
    이 함수가 2단계의 핵심이다.
    도구 호출이 끝날 때까지 반복하는 구조인데
    이게 바로 나중에 배울 LangGraph Agent의 기본 원리야.

    흐름:
    1. 메시지 + 도구 목록 → Gemini 전송
    2. Gemini → 도구 호출 요청 응답
    3. 도구 실행
    4. 결과를 대화에 추가
    5. Gemini에 재전송 → 최종 답변
    6. 최종 답변이 나올 때까지 반복

    Args:
        client      : LLM 클라이언트
        messages    : 현재까지의 대화 히스토리
        system_prompt: 시스템 프롬프트
        counter     : 토큰 카운터

    Returns:
        최종 AI 응답 텍스트
    """
    
    current_messages = messages.copy()
    max_iterations = 5   # 무한 루프 방지 (실무에서는 필수.)
    
    for iteration in range(max_iterations):
        
        # ── Step 1: Gemini에 요청 ──────────────────
        response = client.chat(
            messages=current_messages,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1000,
            stream=False,
            tools=ALL_TOOLS
        )

        # ── Step 2: 도구 호출 여부 확인 ───────────
        if not client.has_tool_call(response):
            # 도구 호출 없음 → 최종 텍스트 응답
            final_text = response.text
            input_tokens  = sum(
                len(m["parts"][0]["text"]) // 4
                for m in current_messages
            )
            output_tokens = len(final_text) // 4
            counter.update_usage(input_tokens, output_tokens)
            return final_text

        # ── Step 3: 도구 정보 추출 ────────────────
        tool_name, tool_args = client.get_tool_call(response)
        print(f"\n   🔧 도구 호출: {tool_name}({tool_args})")

        # ── Step 4: 도구 실행 ─────────────────────
        tool_result = execute_tool(tool_name, tool_args)
        print(f"   📦 도구 결과: {json.dumps(tool_result, ensure_ascii=False)}")

        # ── Step 5: 대화에 도구 호출/결과 추가 ────
        # Gemini Function Calling 프로토콜:
        # AI의 도구 호출 요청과 실행 결과를 대화 히스토리에 추가해야
        # 다음 요청에서 AI가 결과를 참고할 수 있다.

        # AI의 도구 호출 요청을 히스토리에 추가
        current_messages.append({
            "role": "model",
            "parts": [{
                "function_call": {
                    "name": tool_name,
                    "args": tool_args
                }
            }]
        })

        # 도구 실행 결과를 히스토리에 추가
        current_messages.append({
            "role": "user",
            "parts": [{
                "function_response": {
                    "name": tool_name,
                    "response": tool_result
                }
            }]
        })

        # ── Step 6: 다음 반복으로 → 최종 답변 요청

    # max_iterations 초과 시 (비정상 상황)
    return "죄송합니다. 도구 실행 중 문제가 발생했습니다."

def main() -> None:
    """메인 실행 함수"""

    print_welcome()

    # ─────────────────────────────────────
    # 핵심 컴포넌트 초기화
    # ─────────────────────────────────────

    MODEL = "models/gemini-2.5-flash"

    # 1. LLM 클라이언트
    client = LLMClient(model=MODEL)

    # 2. 대화 관리자
    conversation = ConversationManager(
        system_prompt=LLM_MENTOR,
        max_history=20
    )

    # 3. 토큰 카운터
    counter = TokenCounter(model=MODEL)

    print("─" * 44)

    # ─────────────────────────────────────
    # 메인 대화 루프
    # ─────────────────────────────────────

    while True:
        try:
            user_input = input("\n👤 나: ").strip()

            # 빈 입력 무시
            if not user_input:
                continue

            # 커맨드 처리
            if user_input.startswith("/"):
                should_continue = handle_command(user_input, conversation, counter)
                if not should_continue:
                    break
                continue

            # ── LLM 응답 생성 ────────────────────

            # 1. 사용자 메시지를 히스토리에 추가
            conversation.add_user_message(user_input)

            # 2. API 전송용 메시지 목록 준비
            messages = conversation.get_messages_for_api()

            # 3. Function Calling 루프 실행
            #    도구가 필요하면 자동으로 실행하고 최종 답변 반환
            print("\n🤖 AI: ", end="", flush=True)

            full_response = run_function_calling_loop(
                client=client,
                messages=messages,
                system_prompt=conversation.get_system_prompt(),
                counter=counter
            )
            
            print(full_response)

            # 4. AI 응답 히스토리에 추가
            conversation.add_assistant_message(full_response)
            
            

            print()  # 줄바꿈

            # 5. AI 응답을 히스토리에 추가
            conversation.add_assistant_message(full_response)

            # 6. 토큰 사용량 업데이트 (근사값)
            input_tokens  = sum(len(m["parts"][0]["text"]) // 4 for m in messages)
            output_tokens = len(full_response) // 4
            request_cost  = counter.update_usage(input_tokens, output_tokens)

            if request_cost > 0.000001:
                print(
                    f"   💰 예상 비용: "
                    f"${request_cost:.6f} "
                    f"(₩{request_cost * 1380:.4f})"
                )

        except KeyboardInterrupt:
            print("\n\n⚠️  Ctrl+C 감지 — /quit으로 정상 종료를 권장합니다.")
            counter.print_stats()
            break

        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            print("다시 시도하거나 /quit으로 종료하세요.")


if __name__ == "__main__":
    main()