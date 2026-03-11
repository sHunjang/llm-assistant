"""
LangChain Memory 기반 대화 모듈

실무 포인트:
LangChain Memory의 종류와 언제 쓰는지 이해하는 게 핵심이다.

Memory 종류:
─────────────────────────────────────────────
ConversationBufferMemory
  → 대화 전체를 그대로 저장
  → 단점: 대화가 길어질수록 토큰 폭발
  → 적합: 짧은 대화, 테스트용

ConversationBufferWindowMemory
  → 최근 K개 대화만 유지
  → 우리가 1단계에서 max_history로 구현한 것과 동일
  → 적합: 일반 챗봇 서비스

ConversationSummaryMemory
  → 오래된 대화를 LLM으로 요약해서 압축
  → 토큰 절약 + 맥락 유지
  → 단점: 요약 API 호출 비용 추가
  → 적합: 장기 대화가 필요한 서비스

ConversationSummaryBufferMemory
  → 최근 대화는 그대로, 오래된 건 요약
  → 위 두 방식의 장점 결합
  → 실무 프로덕션 환경에서 가장 많이 사용
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import (
    ConversationBufferWindowMemory,
    ConversationSummaryBufferMemory,
)
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()


class WindowMemoryChat:
    """
    최근 K개 대화만 유지하는 채팅 클래스

    실무 포인트:
    1단계에서 max_history=20으로 직접 구현한 것과 동일한 개념.
    LangChain은 ConversationBufferWindowMemory로 제공.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        system_prompt: str = "당신은 친절한 AI 어시스턴트입니다. 한국어로 답변하세요.",
        window_size: int = 5,       # 최근 5개 대화만 유지
        temperature: float = 0.7,
    ):
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
        )

        # window_size: 유지할 최근 대화 수
        # k=5 → 사용자 5개 + AI 5개 = 총 10개 메시지 유지
        self.memory = ConversationBufferWindowMemory(
            k=window_size,
            return_messages=True,
            memory_key="chat_history"
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        self.chain = (
            RunnablePassthrough.assign(
                chat_history=lambda x: self.memory.load_memory_variables({})["chat_history"]
            )
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        print(f"✅ Window Memory Chat 초기화 (최근 {window_size}개 대화 유지)")

    def chat(self, user_input: str) -> str:
        response = self.chain.invoke({"input": user_input})
        self.memory.save_context(
            {"input": user_input},
            {"output": response}
        )
        return response

    def get_memory_size(self) -> int:
        """현재 메모리에 저장된 메시지 수"""
        return len(self.memory.load_memory_variables({})["chat_history"])

    def clear(self) -> None:
        self.memory.clear()
        print("🗑️  메모리 초기화 완료")


class SummaryMemoryChat:
    """
    오래된 대화를 요약하는 채팅 클래스

    실무 포인트:
    ConversationSummaryBufferMemory:
    - max_token_limit 이하: 그대로 유지
    - max_token_limit 초과: LLM으로 요약 후 압축

    장점: 토큰 폭발 방지 + 오래된 맥락도 유지
    단점: 요약을 위한 추가 LLM 호출 발생
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        system_prompt: str = "당신은 친절한 AI 어시스턴트입니다. 한국어로 답변하세요.",
        max_token_limit: int = 500,   # 이 토큰 초과 시 요약
        temperature: float = 0.7,
    ):
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
        )

        # max_token_limit 초과 시 자동으로 요약 생성
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,               # 요약에 사용할 LLM
            max_token_limit=max_token_limit,
            return_messages=True,
            memory_key="chat_history"
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        self.chain = (
            RunnablePassthrough.assign(
                chat_history=lambda x: self.memory.load_memory_variables({})["chat_history"]
            )
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        print(
            f"✅ Summary Memory Chat 초기화 "
            f"(토큰 제한: {max_token_limit})"
        )

    def chat(self, user_input: str) -> str:
        response = self.chain.invoke({"input": user_input})
        self.memory.save_context(
            {"input": user_input},
            {"output": response}
        )
        return response

    def get_summary(self) -> str:
        """현재 저장된 대화 요약 반환"""
        messages = self.memory.load_memory_variables({})["chat_history"]
        # SystemMessage가 요약 내용을 담고 있음
        from langchain_core.messages import SystemMessage
        for msg in messages:
            if isinstance(msg, SystemMessage):
                return msg.content
        return "요약 없음 (대화가 충분하지 않음)"

    def clear(self) -> None:
        self.memory.clear()
        print("🗑️  메모리 초기화 완료")