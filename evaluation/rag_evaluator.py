"""
RAG 품질 평가 모듈

실무 포인트:
RAG 시스템을 만들고 나서
"잘 동작하고 있나?" 를 어떻게 측정할까?

주요 평가 지표:
─────────────────────────────────────────
1. 답변 관련성 (Answer Relevancy)
   질문과 답변이 얼마나 관련 있는가?
   → LLM으로 0~1 점수 측정

2. 충실도 (Faithfulness)
   답변이 검색된 문서 내용에 충실한가?
   → 문서에 없는 내용을 지어내지 않는가?

3. 문서 관련성 (Context Relevancy)
   검색된 문서가 질문과 얼마나 관련 있는가?
   → Retriever 품질 측정

실무에서 이 지표들이 중요한 이유:
- 답변 관련성 낮음 → 프롬프트 개선 필요
- 충실도 낮음      → 환각 발생, 위험!
- 문서 관련성 낮음 → 청킹/임베딩 전략 개선 필요
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dataclasses import dataclass, field
from typing import Optional
import re
from core.config import get_llm_config
from core.logger import LLMLogger

logger = LLMLogger(__name__)


@dataclass
class EvaluationResult:
    """단일 평가 결과"""
    question: str
    answer: str
    contexts: list[str]
    relevancy_score: float = 0.0       # 답변 관련성
    faithfulness_score: float = 0.0    # 충실도
    context_score: float = 0.0         # 문서 관련성
    overall_score: float = 0.0         # 종합 점수
    feedback: str = ""                 # 평가 피드백


@dataclass
class EvaluationReport:
    """전체 평가 리포트"""
    results: list[EvaluationResult] = field(default_factory=list)
    avg_relevancy: float = 0.0
    avg_faithfulness: float = 0.0
    avg_context: float = 0.0
    avg_overall: float = 0.0
    total_questions: int = 0

    def summary(self) -> str:
        """리포트 요약 문자열 반환"""
        return (
            f"\n{'='*50}\n"
            f"📊 RAG 평가 리포트\n"
            f"{'='*50}\n"
            f"총 질문 수      : {self.total_questions}개\n"
            f"답변 관련성     : {self.avg_relevancy:.2f} / 1.0\n"
            f"충실도          : {self.avg_faithfulness:.2f} / 1.0\n"
            f"문서 관련성     : {self.avg_context:.2f} / 1.0\n"
            f"{'─'*50}\n"
            f"종합 점수       : {self.avg_overall:.2f} / 1.0\n"
            f"{'='*50}"
        )


class RAGEvaluator:
    """
    LLM 기반 RAG 품질 평가기

    실무 포인트:
    LLM으로 LLM 결과를 평가하는 방식 (LLM-as-a-Judge).
    완벽하지는 않지만 빠르게 품질을 측정할 수 있어.
    더 정교한 평가는 RAGAS 라이브러리를 사용해.
    """

    def __init__(self):
        config = get_llm_config()
        self.llm = ChatGoogleGenerativeAI(
            model=config.default_model,
            temperature=0.0,    # 평가는 일관성이 중요 → temperature=0
        )
        self._build_eval_chains()
        logger.info("RAG 평가기 초기화 완료")

    def _build_eval_chains(self) -> None:
        """평가용 Chain 3개 구성"""

        # 1. 답변 관련성 평가 Chain
        relevancy_prompt = ChatPromptTemplate.from_template("""
질문과 답변을 보고 답변이 질문에 얼마나 관련 있는지 평가하세요.

질문: {question}
답변: {answer}

평가 기준:
- 1.0: 질문에 완벽하게 답변
- 0.7: 대체로 관련 있지만 일부 누락
- 0.4: 부분적으로만 관련
- 0.0: 전혀 관련 없음

숫자만 반환하세요 (예: 0.8)
""")
        self.relevancy_chain = (
            relevancy_prompt | self.llm | StrOutputParser()
        )

        # 2. 충실도 평가 Chain
        faithfulness_prompt = ChatPromptTemplate.from_template("""
아래 문서 내용을 기반으로 답변이 문서에 충실한지 평가하세요.
문서에 없는 내용을 지어냈다면 낮은 점수를 주세요.

참고 문서:
{contexts}

답변: {answer}

평가 기준:
- 1.0: 모든 내용이 문서에 근거함
- 0.7: 대부분 문서 기반, 일부 추론 포함
- 0.4: 문서 기반과 지어낸 내용이 섞임
- 0.0: 문서와 무관한 내용

숫자만 반환하세요 (예: 0.9)
""")
        self.faithfulness_chain = (
            faithfulness_prompt | self.llm | StrOutputParser()
        )

        # 3. 문서 관련성 평가 Chain
        context_prompt = ChatPromptTemplate.from_template("""
질문에 대해 검색된 문서가 얼마나 관련 있는지 평가하세요.

질문: {question}

검색된 문서:
{contexts}

평가 기준:
- 1.0: 문서가 질문에 완벽하게 관련
- 0.7: 대체로 관련 있음
- 0.4: 부분적으로만 관련
- 0.0: 전혀 관련 없음

숫자만 반환하세요 (예: 0.7)
""")
        self.context_chain = (
            context_prompt | self.llm | StrOutputParser()
        )

    def _parse_score(self, raw: str) -> float:
        """LLM 응답에서 점수 파싱"""
        try:
            numbers = re.findall(r'\d+\.?\d*', raw.strip())
            if numbers:
                score = float(numbers[0])
                return min(max(score, 0.0), 1.0)
        except Exception:
            pass
        return 0.5  # 파싱 실패 시 기본값

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: list[str]
    ) -> EvaluationResult:
        """
        단일 QA 쌍 평가

        Args:
            question: 질문
            answer: RAG 시스템의 답변
            contexts: 검색된 문서 목록

        Returns:
            EvaluationResult
        """
        context_text = "\n\n".join(contexts)

        # 3가지 지표 평가
        relevancy_raw = self.relevancy_chain.invoke({
            "question": question,
            "answer": answer
        })
        faithfulness_raw = self.faithfulness_chain.invoke({
            "contexts": context_text,
            "answer": answer
        })
        context_raw = self.context_chain.invoke({
            "question": question,
            "contexts": context_text
        })

        # 점수 파싱
        relevancy = self._parse_score(relevancy_raw)
        faithfulness = self._parse_score(faithfulness_raw)
        context = self._parse_score(context_raw)
        overall = (relevancy + faithfulness + context) / 3

        result = EvaluationResult(
            question=question,
            answer=answer,
            contexts=contexts,
            relevancy_score=relevancy,
            faithfulness_score=faithfulness,
            context_score=context,
            overall_score=overall
        )

        logger.info(
            "평가 완료",
            relevancy=relevancy,
            faithfulness=faithfulness,
            context=context,
            overall=round(overall, 3)
        )

        return result

    def evaluate_batch(
        self,
        test_cases: list[dict]
    ) -> EvaluationReport:
        """
        다수의 QA 쌍 일괄 평가

        Args:
            test_cases: [{"question": ..., "answer": ..., "contexts": [...]}, ...]

        Returns:
            EvaluationReport
        """
        report = EvaluationReport()

        for i, case in enumerate(test_cases, 1):
            print(f"   평가 중... ({i}/{len(test_cases)})", end="\r")
            result = self.evaluate_single(
                question=case["question"],
                answer=case["answer"],
                contexts=case.get("contexts", [])
            )
            report.results.append(result)

        # 평균 계산
        n = len(report.results)
        if n > 0:
            report.total_questions = n
            report.avg_relevancy = sum(
                r.relevancy_score for r in report.results
            ) / n
            report.avg_faithfulness = sum(
                r.faithfulness_score for r in report.results
            ) / n
            report.avg_context = sum(
                r.context_score for r in report.results
            ) / n
            report.avg_overall = sum(
                r.overall_score for r in report.results
            ) / n

        return report