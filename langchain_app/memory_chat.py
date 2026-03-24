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

ConversationBufferWindowMemory → 직접 슬라이싱으로 구현

RunnableWithMessageHistory + 요약 Chain

ConversationSummaryBufferMemory
  → 최근 대화는 그대로, 오래된 건 요약
  → 위 두 방식의 장점 결합
  → 실무 프로덕션 환경에서 가장 많이 사용
"""

from core.llm_factory import create_llm
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv

load_dotenv()


class WindowMemoryChat:
    """
    최근 K개 대화만 유지하는 채팅 클래스

    실무 포인트:
    window_size로 토큰 사용량을 직접 제어할 수 있다.
    일반 챗봇 서비스에서 가장 많이 쓰는 방식.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        system_prompt: str = "당신은 친절한 AI 어시스턴트입니다. 한국어로 답변하세요.",
        window_size: int = 5,       # 최근 5개 대화만 유지
        temperature: float = 0.7,
    ):
        
        self.window_size = window_size
        self.history = ChatMessageHistory()
        
        self.llm = create_llm(
            model=model,
            temperature=temperature,
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

        print(f"✅ Window Memory Chat 초기화 (최근 {window_size}개 대화 유지)")

    def _get_window_history(self) -> list:
        """최근 window_size * 2개 메시지만 반환 (human + ai 쌍)"""
        messages = self.history.messages
        # window_size개 대화쌍 = window_size * 2개 메시지
        return messages[-(self.window_size * 2):]

    def chat(self, user_input: str) -> str:
        """윈도우 히스토리 기반 응답 생성"""
        response = self.chain.invoke({
            "input": user_input,
            "chat_history": self._get_window_history()
        })

        # 히스토리에 저장
        self.history.add_user_message(user_input)
        self.history.add_ai_message(response)

        return response

    def get_memory_size(self) -> int:
        """현재 전체 히스토리 메시지 수"""
        return len(self.history.messages)

    def get_window_size(self) -> int:
        """현재 LLM에 전달되는 메시지 수"""
        return len(self._get_window_history())

    def clear(self) -> None:
        self.history.clear()
        print("🗑️  메모리 초기화 완료")


class SummaryMemoryChat:
    """
    오래된 대화를 요약하는 채팅 클래스

    동작 방식:
    1. 대화가 max_turns를 초과하면
    2. 오래된 대화를 LLM으로 요약
    3. 요약본 + 최근 대화만 유지

    실무 포인트:
    요약 시 추가 LLM 호출이 발생한다.
    비용 vs 맥락 유지를 트레이드오프로 고려해야 함.

    장점: 토큰 폭발 방지 + 오래된 맥락도 유지
    단점: 요약을 위한 추가 LLM 호출 발생
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        system_prompt: str = "당신은 친절한 AI 어시스턴트입니다. 한국어로 답변하세요.",
        max_turns: int = 6,         # 이 대화 수 초과 시 요약
        recent_turns: int = 3,      # 요약 후 유지할 최근 대화 수
        temperature: float = 0.7,
    ):
        self.max_turns = max_turns
        self.recent_turns = recent_turns
        self.summary = ""           # 누적 요약본
        self.history = ChatMessageHistory()

        self.llm = create_llm(
            model=model,
            temperature=temperature,
        )

        # 메인 대화 Chain
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()


        # 요약 Chain
        self.summary_prompt = ChatPromptTemplate.from_template("""
이전 대화 요약과 새로운 대화를 합쳐서 간결하게 요약해주세요.

이전 요약:
{previous_summary}

새로운 대화:
{new_conversations}

요약 규칙:
- 핵심 정보만 유지 (이름, 중요 사실, 결정사항)
- 3-5문장으로 압축
- 한국어로 작성
""")
        self.summary_chain = self.summary_prompt | self.llm | StrOutputParser()

        print(
            f"✅ Summary Memory Chat 초기화 "
            f"(최대 {max_turns}턴, 요약 후 {recent_turns}턴 유지)"
        )

    def _build_context(self) -> list:
        """현재 컨텍스트 구성 (요약 + 최근 대화)"""
        messages = []
        from langchain_core.messages import SystemMessage

        # 요약본이 있으면 시스템 메시지로 추가
        if self.summary:
            messages.append(SystemMessage(
                content=f"이전 대화 요약:\n{self.summary}"
            ))

        # 최근 대화 추가
        recent = self.history.messages[-(self.recent_turns * 2):]
        messages.extend(recent)

        return messages

    def _compress_if_needed(self) -> None:
        """대화가 max_turns 초과 시 요약으로 압축"""
        total_turns = len(self.history.messages) // 2

        if total_turns <= self.max_turns:
            return

        # 오래된 대화 추출 (최근 recent_turns 제외)
        old_messages = self.history.messages[:-(self.recent_turns * 2)]

        if not old_messages:
            return

        # 오래된 대화를 텍스트로 변환
        conv_text = "\n".join([
            f"{'사용자' if m.type == 'human' else 'AI'}: {m.content}"
            for m in old_messages
        ])

        # 요약 생성
        print("\n📝 대화 요약 중...")
        self.summary = self.summary_chain.invoke({
            "previous_summary": self.summary or "없음",
            "new_conversations": conv_text
        })

        # 히스토리를 최근 대화만 남기도록 교체
        recent_messages = self.history.messages[-(self.recent_turns * 2):]
        self.history.clear()
        for msg in recent_messages:
            if msg.type == "human":
                self.history.add_user_message(msg.content)
            else:
                self.history.add_ai_message(msg.content)

        print(f"✅ 요약 완료 (히스토리 압축됨)\n")

    def chat(self, user_input: str) -> str:
        """요약 메모리 기반 응답 생성"""

        # 필요 시 압축
        self._compress_if_needed()

        # 컨텍스트 구성 후 응답 생성
        response = self.chain.invoke({
            "input": user_input,
            "chat_history": self._build_context()
        })

        # 히스토리 저장
        self.history.add_user_message(user_input)
        self.history.add_ai_message(response)

        return response

    def get_summary(self) -> str:
        """현재 요약본 반환"""
        return self.summary if self.summary else "요약 없음 (대화가 충분하지 않음)"

    def clear(self) -> None:
        self.history.clear()
        self.summary = ""
        print("🗑️  메모리 초기화 완료")
