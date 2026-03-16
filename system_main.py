"""
6단계 시스템 엔지니어링 메인 실행 파일

학습 포인트:
- 설정 관리가 어떻게 동작하는지
- 로깅이 어디에 기록되는지
- 캐싱 효과 (같은 질문 두 번)
- RAG 평가 점수 해석
"""

import time
import os
from dotenv import load_dotenv

from core.config import get_llm_config, get_rag_config, get_app_config
from core.logger import LLMLogger
from core.cache import ResponseCache
from core.exceptions import LLMRateLimitError, ConfigError
from evaluation.rag_evaluator import RAGEvaluator

from langchain_app.rag_chain import LangChainRAG

load_dotenv()
logger = LLMLogger(__name__)


def demo_config() -> None:
    """설정 관리 데모"""
    print("\n📌 [1] 설정 관리 데모")
    print("─" * 40)

    llm_config = get_llm_config()
    rag_config = get_rag_config()
    app_config = get_app_config()

    print(f"LLM 모델     : {llm_config.default_model}")
    print(f"Temperature  : {llm_config.temperature}")
    print(f"Chunk Size   : {rag_config.chunk_size}")
    print(f"Top K        : {rag_config.top_k}")
    print(f"환경         : {app_config.environment}")
    print(f"로그 레벨    : {app_config.log_level}")
    print(f"캐시 활성화  : {app_config.cache_enabled}")
    print(f"\n✅ 모든 설정이 .env에서 자동 로딩됨")
    print(f"✅ 로그 파일  : {app_config.log_file}")


def demo_logging() -> None:
    """로깅 데모"""
    print("\n📌 [2] 구조화된 로깅 데모")
    print("─" * 40)

    demo_logger = LLMLogger("demo")

    demo_logger.info("애플리케이션 시작")
    demo_logger.log_llm_call(
        model="gemini-2.0-flash-lite",
        input_tokens=150,
        output_tokens=200,
        latency_ms=823.4,
        success=True
    )
    demo_logger.log_rag_search(
        query="테스트 질문",
        num_results=3,
        top_score=0.85,
        latency_ms=45.2
    )
    demo_logger.warning("Rate Limit 임박", remaining_calls=5)

    config = get_app_config()
    print(f"\n✅ 로그가 콘솔 + {config.log_file} 에 동시 기록됨")


def demo_cache() -> None:
    """캐싱 데모"""
    print("\n📌 [3] 응답 캐싱 데모")
    print("─" * 40)

    cache = ResponseCache()
    test_prompt = "LangChain이 뭐야?"
    test_response = "LangChain은 LLM 기반 애플리케이션 프레임워크입니다."

    # 첫 번째 요청 (캐시 미스)
    start = time.time()
    result = cache.get(test_prompt)
    elapsed = (time.time() - start) * 1000
    print(f"1차 조회 (캐시 미스): {result} | {elapsed:.2f}ms")

    # 캐시 저장
    cache.set(test_prompt, test_response)

    # 두 번째 요청 (캐시 히트)
    start = time.time()
    result = cache.get(test_prompt)
    elapsed = (time.time() - start) * 1000
    print(f"2차 조회 (캐시 히트): {result[:20]}... | {elapsed:.2f}ms")

    stats = cache.get_stats()
    print(f"\n캐시 통계: {stats}")
    print(f"✅ 캐시 히트 시 API 호출 없음 → 비용 0 + 응답 속도 향상")


def demo_exceptions() -> None:
    """예외 처리 데모"""
    print("\n📌 [4] 예외 처리 데모")
    print("─" * 40)

    # RateLimitError 처리
    try:
        raise LLMRateLimitError(
            "일일 요청 한도를 초과했습니다.",
            retry_after=43
        )
    except LLMRateLimitError as e:
        print(f"RateLimitError 감지: {e}")
        print(f"  → {e.retry_after}초 후 재시도 필요")
        logger.log_error("rate_limit_exceeded", e)

    # ConfigError 처리
    try:
        raise ConfigError(
            "GEMINI_API_KEY가 설정되지 않았습니다.",
            {"env_var": "GEMINI_API_KEY"}
        )
    except ConfigError as e:
        print(f"\nConfigError 감지: {e}")
        logger.log_error("config_missing", e)

    print(f"\n✅ 에러 종류별 다른 처리 로직 적용 가능")


def demo_rag_evaluation() -> None:
    """RAG 평가 데모"""
    print("\n📌 [5] RAG 품질 평가 데모")
    print("─" * 40)

    pdf_path = "data/researchPaper.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ {pdf_path} 파일이 없어서 평가를 건너뜁니다.")
        return

    # RAG 시스템 구성
    print("📄 문서 인덱싱 중...")
    rag = LangChainRAG()
    rag.index_document(pdf_path)

    # 테스트 질문 3개
    test_questions = [
        "이 논문에서 사용한 AI 모델이 뭐야?",
        "실험 환경이 어떻게 구성됐어?",
        "연구의 한계점이 뭐야?",
    ]

    # 테스트 케이스 구성
    print("\n🔍 테스트 질문에 대한 답변 생성 중...")
    test_cases = []
    for q in test_questions:
        answer = rag.ask(q)
        # 실제로는 retriever에서 contexts를 가져와야 함
        # 여기서는 답변만으로 간단히 평가
        test_cases.append({
            "question": q,
            "answer": answer,
            "contexts": [answer]  # 간소화
        })
        print(f"   Q: {q[:30]}...")
        time.sleep(1)  # Rate Limit 방지

    # 평가 실행
    print("\n📊 품질 평가 중...")
    evaluator = RAGEvaluator()
    report = evaluator.evaluate_batch(test_cases)

    # 결과 출력
    print(report.summary())

    # 개별 결과 출력
    print("\n📋 개별 평가 결과:")
    for i, result in enumerate(report.results, 1):
        print(f"\n  [{i}] {result.question[:40]}")
        print(f"      관련성: {result.relevancy_score:.2f} | "
              f"충실도: {result.faithfulness_score:.2f} | "
              f"문서관련성: {result.context_score:.2f}")


def main() -> None:
    print("""
╔══════════════════════════════════════════╗
║     ⚙️  LLM 시스템 엔지니어링             ║
║     Powered by Gemini  v6.0             ║
╚══════════════════════════════════════════╝
""")

    print("실행할 데모를 선택하세요:")
    print("1. 설정 관리     (Config)")
    print("2. 구조화 로깅   (Logger)")
    print("3. 응답 캐싱     (Cache)")
    print("4. 예외 처리     (Exception)")
    print("5. RAG 품질 평가 (Evaluator)")
    print("6. 전체 실행")

    choice = input("\n선택 (1~6): ").strip()

    if choice == "1":
        demo_config()
    elif choice == "2":
        demo_logging()
    elif choice == "3":
        demo_cache()
    elif choice == "4":
        demo_exceptions()
    elif choice == "5":
        demo_rag_evaluation()
    elif choice == "6":
        demo_config()
        demo_logging()
        demo_cache()
        demo_exceptions()
        demo_rag_evaluation()
    else:
        print("1~6 중에서 선택해줘.")


if __name__ == "__main__":
    main()