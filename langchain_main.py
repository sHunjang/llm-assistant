"""
LangChain 애플리케이션 메인 실행 파일

메뉴:
1. 기본 채팅        → LangChain LCEL Chain
2. RAG 채팅         → LangChain RAG Chain
3. Memory 채팅 비교 → Window vs Summary Memory
"""

from langchain_app.chat import LangChainChat
from langchain_app.rag_chain import LangChainRAG
from langchain_app.memory_chat import WindowMemoryChat, SummaryMemoryChat
from prompts.system_prompts import LLM_MENTOR


def print_welcome() -> None:
    print("""
╔══════════════════════════════════════════╗
║     🦜 LangChain AI 어시스턴트            ║
║     Powered by Gemini 2.5 Flash  v4.0   ║
╚══════════════════════════════════════════╝
""")


def run_basic_chat() -> None:
    """
    기본 채팅 모드

    학습 포인트:
    - LCEL Chain 동작 방식
    - LangChain Memory 자동 관리
    - 스트리밍 응답
    """

    print("\n📌 모드: 기본 채팅 (LCEL Chain)")
    print("커맨드: /clear → 초기화, /history → 히스토리, /quit → 종료\n")

    chat = LangChainChat(
        system_prompt=LLM_MENTOR,
        temperature=0.7
    )

    while True:
        user_input = input("\n👤 나: ").strip()
        if not user_input:
            continue

        if user_input == "/clear":
            chat.clear_memory()
            continue
        elif user_input == "/history":
            history = chat.get_history()
            print(f"\n💬 대화 히스토리 ({len(history)}개 메시지):")
            for msg in history:
                role = "👤" if msg.type == "human" else "🤖"
                print(f"   {role} {msg.content[:50]}...")
            continue
        elif user_input == "/quit":
            break

        # 스트리밍 응답
        print("\n🤖 AI: ", end="", flush=True)
        for chunk in chat.stream_chat(user_input):
            print(chunk, end="", flush=True)
        print()


def run_rag_chat() -> None:
    """
    RAG 채팅 모드

    학습 포인트:
    - LangChain RAG Chain 동작 방식
    - 3단계 직접 구현과 코드량 비교
    - GoogleGenerativeAIEmbeddings 활용
    """

    print("\n📌 모드: RAG 채팅 (LangChain RAG)")
    print("커맨드: /quit → 종료\n")

    rag = LangChainRAG(top_k=3)

    # PDF 파일 경로 입력
    while True:
        path = input("📄 PDF 파일 경로: ").strip().strip('"').strip("'")
        from pathlib import Path
        if Path(path).exists() and path.lower().endswith(".pdf"):
            break
        print("❌ 유효한 PDF 파일 경로를 입력해주세요.")

    # 문서 인덱싱
    rag.index_document(path)

    print("\n✅ 준비 완료! 문서에 대해 질문해보세요.\n")

    while True:
        question = input("\n👤 질문: ").strip()
        if not question:
            continue
        if question == "/quit":
            break

        print("\n🤖 AI: ", end="", flush=True)
        for chunk in rag.stream_ask(question):
            print(chunk, end="", flush=True)
        print()


def run_memory_comparison() -> None:
    """
    Memory 방식 비교 모드

    학습 포인트:
    - Window Memory vs Summary Memory 차이
    - 실무에서 언제 어떤 Memory를 쓰는지
    """

    print("\n📌 모드: Memory 방식 비교")
    print("같은 질문을 두 방식으로 테스트해봐.\n")
    print("Window Memory  → 최근 5개 대화만 유지")
    print("Summary Memory → 오래된 대화를 자동 요약\n")
    print("커맨드: /summary → 요약 확인, /quit → 종료\n")

    window_chat   = WindowMemoryChat(window_size=5)
    summary_chat  = SummaryMemoryChat(max_token_limit=300)

    while True:
        user_input = input("\n👤 나: ").strip()
        if not user_input:
            continue

        if user_input == "/summary":
            print(f"\n📝 Summary Memory 요약:\n{summary_chat.get_summary()}")
            print(f"💬 Window Memory 메시지 수: {window_chat.get_memory_size()}개")
            continue
        elif user_input == "/quit":
            break

        # Window Memory 응답
        print("\n[Window Memory]")
        print("🤖 AI: ", end="", flush=True)
        response_w = window_chat.chat(user_input)
        print(response_w)

        # Summary Memory 응답
        print("\n[Summary Memory]")
        print("🤖 AI: ", end="", flush=True)
        response_s = summary_chat.chat(user_input)
        print(response_s)


def main() -> None:
    print_welcome()

    # 모드 선택
    print("실행할 모드를 선택하세요:")
    print("1. 기본 채팅     (LCEL Chain)")
    print("2. RAG 채팅      (LangChain RAG)")
    print("3. Memory 비교   (Window vs Summary)")

    while True:
        choice = input("\n선택 (1/2/3): ").strip()
        if choice == "1":
            run_basic_chat()
            break
        elif choice == "2":
            run_rag_chat()
            break
        elif choice == "3":
            run_memory_comparison()
            break
        else:
            print("1, 2, 3 중에서 선택해줘.")


if __name__ == "__main__":
    main()