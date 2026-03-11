"""
RAG 기반 PDF 검색 챗봇 실행 파일

실행 방법:
    python rag_main.py

사용 흐름:
    1. PDF 파일 경로 입력
    2. 문서 인덱싱 (최초 1회)
    3. 질문 입력
    4. AI가 문서 기반으로 답변
"""

from assistant.client import LLMClient
from assistant.token_counter import TokenCounter
from rag.retriever import RAGRetriever
from prompts.system_prompts import RAG_QA


def print_welcome() -> None:
    print("""
╔══════════════════════════════════════════╗
║     📚 PDF 기반 RAG 검색 챗봇              ║
║     Powered by Gemini 2.5 Flash  v3.0    ║
╚══════════════════════════════════════════╝

사용 흐름:
   1. PDF 파일 경로 입력
   2. 문서 인덱싱 (자동)
   3. 질문 입력 → AI가 문서 기반으로 답변

💡 커맨드:
   /stats  → 시스템 현황
   /quit   → 종료
""")


def get_pdf_path() -> str:
    """PDF 파일 경로 입력 받기"""
    while True:
        path = input("📄 PDF 파일 경로를 입력하세요: ").strip()

        if not path:
            print("❌ 경로를 입력해주세요.")
            continue

        # 따옴표 제거 (드래그앤드롭 시 따옴표가 붙는 경우)
        path = path.strip('"').strip("'")

        from pathlib import Path
        if not Path(path).exists():
            print(f"❌ 파일을 찾을 수 없습니다: {path}")
            continue

        if not path.lower().endswith(".pdf"):
            print("❌ PDF 파일만 지원합니다.")
            continue

        return path


def build_rag_prompt(context: str, question: str) -> str:
    """
    검색된 문서 컨텍스트와 질문을 조합해서
    LLM에 전달할 최종 프롬프트 생성

    실무 포인트:
    이 프롬프트 구조가 RAG 답변 품질의 핵심이다.
    컨텍스트를 어떻게 구조화하느냐에 따라 답변이 달라진다.
    """
    return f"""다음 문서 내용을 참고해서 질문에 답변해주세요.

=== 참고 문서 ===
{context}

=== 질문 ===
{question}
"""


def main() -> None:
    print_welcome()

    MODEL = "models/gemini-2.5-flash"

    # 컴포넌트 초기화
    client   = LLMClient(model=MODEL)
    counter  = TokenCounter(model=MODEL)
    retriever = RAGRetriever(
        chunk_size=500,
        chunk_overlap=50,
        top_k=3
    )

    # PDF 로딩 및 인덱싱
    pdf_path = get_pdf_path()
    retriever.index_document(pdf_path)

    print("─" * 44)
    print("✅ 준비 완료! 문서에 대해 질문해보세요.\n")

    # 대화 루프
    while True:
        try:
            user_input = input("\n👤 질문: ").strip()

            if not user_input:
                continue

            # 커맨드 처리
            if user_input == "/stats":
                stats = retriever.get_stats()
                print(f"\n📊 RAG 시스템 현황")
                print(f"   저장된 청크 수  : {stats['total_chunks']}개")
                print(f"   검색 결과 수    : {stats['top_k']}개")
                print(f"   임베딩 모델     : {stats['embedding_model']}")
                counter.print_stats()
                continue

            if user_input in ["/quit", "/q"]:
                print("\n📊 최종 사용량:")
                counter.print_stats()
                print("\n👋 수고하셨습니다!")
                break

            # ── RAG 파이프라인 실행 ───────────────

            # Step 1: 관련 문서 검색
            print("\n🔍 관련 문서 검색 중...")
            search_results = retriever.retrieve(user_input)

            # Step 2: 검색 결과 출력 (투명성)
            print(f"   📎 검색된 문서 {len(search_results)}개:")
            for i, r in enumerate(search_results, 1):
                meta = r["metadata"]
                print(
                    f"      [{i}] {meta.get('source')} "
                    f"{meta.get('page_num')}페이지 "
                    f"(관련도: {r['similarity']:.1%})"
                )

            # Step 3: 컨텍스트 + 질문 조합
            context = retriever.format_context(search_results)
            prompt  = build_rag_prompt(context, user_input)

            # Step 4: LLM 답변 생성
            print("\n🤖 AI: ", end="", flush=True)

            messages = [{"role": "user", "parts": [{"text": prompt}]}]
            response = client.chat(
                messages=messages,
                system_prompt=RAG_QA,
                temperature=0.3,    # RAG는 낮은 temperature 권장 (사실 기반)
                max_tokens=1500,
                stream=True
            )

            full_response = ""
            for chunk in response:
                print(chunk, end="", flush=True)
                full_response += chunk
            print()

            # Step 5: 토큰 사용량 업데이트
            input_tokens  = len(prompt) // 4
            output_tokens = len(full_response) // 4
            cost = counter.update_usage(input_tokens, output_tokens)

            if cost > 0.000001:
                print(
                    f"   💰 예상 비용: "
                    f"${cost:.6f} (₩{cost * 1380:.4f})"
                )

        except KeyboardInterrupt:
            print("\n\n⚠️  Ctrl+C — /quit으로 정상 종료 권장")
            break

        except Exception as e:
            print(f"\n❌ 오류: {e}")
            print("다시 시도하거나 /quit으로 종료하세요.")


if __name__ == "__main__":
    main()