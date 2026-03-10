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
"""

from assistant.client import LLMClient
from assistant.conversation import ConversationManager
from assistant.token_counter import TokenCounter
from prompts.system_prompts import LLM_MENTOR


def print_welcome() -> None:
    """시작 화면 출력"""
    print("""
╔══════════════════════════════════════════╗
║     🤖 LLM 엔지니어링 AI 멘토            ║
║     Powered by Gemini 2.0 Flash  v1.0   ║
╚══════════════════════════════════════════╝

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


def main() -> None:
    """메인 실행 함수"""

    print_welcome()

    # ─────────────────────────────────────
    # 핵심 컴포넌트 초기화
    # ─────────────────────────────────────

    MODEL = "gemini-2.0-flash"

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

            # 3. 스트리밍 응답 받기
            print("\n🤖 AI: ", end="", flush=True)

            full_response = ""
            stream = client.chat(
                messages=messages,
                system_prompt=conversation.get_system_prompt(),  # 매 요청마다 전달
                temperature=0.7,
                max_tokens=1000,
                stream=True
            )

            # 4. 스트리밍 청크를 실시간으로 출력
            for chunk in stream:
                print(chunk, end="", flush=True)
                full_response += chunk

            print()  # 줄바꿈

            # 5. AI 응답을 히스토리에 추가
            conversation.add_assistant_message(full_response)

            # 6. 토큰 사용량 업데이트 (근사값)
            input_tokens  = sum(len(m["parts"]) // 4 for m in messages)
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