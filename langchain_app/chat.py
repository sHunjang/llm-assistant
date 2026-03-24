"""
LangChain 기본 채팅 모듈

실무 포인트:
1단계에서 직접 구현했던 것들을
LangChain이 어떻게 처리하는지 비교하면서 보면 된다.

직접 구현 vs LangChain 비교:
─────────────────────────────────────────
직접 구현:
  - API 키 설정, 클라이언트 초기화
  - messages 딕셔너리 직접 구성
  - 응답에서 텍스트 직접 파싱
  - 히스토리 직접 관리

LangChain:
  - ChatGoogleGenerativeAI 한 줄로 초기화
  - HumanMessage, AIMessage 객체로 관리
  - response.content로 바로 접근
  - RunnableWithMessageHistory + ChatMessageHistory가 히스토리 관리
"""

from core.llm_factory import create_llm
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv

load_dotenv()


class LangChainChat:
    """
    LangChain 기반 기본 채팅 클래스

    핵심 컴포넌트:
    1. ChatGoogleGenerativeAI → LLM 모델
    2. ChatPromptTemplate     → 프롬프트 템플릿
    3. RunnableWithMessageHistory + ChatMessageHistory → 대화 히스토리
    4. StrOutputParser        → 응답 파싱
    5. LCEL Chain (|)         → 전체 파이프라인 연결
    
    실무 포인트:
    session_id를 사용해서 여러 사용자의 대화를
    독립적으로 관리할 수 있다.
    → 실제 서비스에서 user_id를 session_id로 활용
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        system_prompt: str = "당신은 친절한 AI 어시스턴트입니다. 한국어로 답변하세요.",
        temperature: float = 0.7,
    ):
        # ── 1. LLM 초기화 ────────────────────────
        # 직접 구현: genai.Client(api_key=...) + 모델 설정
        # LangChain: 한 줄로 끝!
        self.llm = create_llm(
            model=model,
            temperature=temperature,
        )

        # ── 2. 프롬프트 템플릿 ───────────────────
        # MessagesPlaceholder: 대화 히스토리가 들어갈 자리
        # 직접 구현: messages 리스트를 수동으로 구성
        # LangChain: 템플릿에 {chat_history} 자리 지정
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        # 세션별 히스토리 저장소
        # { session_id: ChatMessageHistory }
        self.store = {}

        # 기본 Chain (히스토리 없음)
        base_chain = self.prompt | self.llm | StrOutputParser()

        # RunnableWithMessageHistory로 히스토리 자동 관리
        self.chain = RunnableWithMessageHistory(
            base_chain,
            self._get_session_history,      # 세션 히스토리 조회 함수
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        self.session_id = "default"
        print(f"✅ LangChain Chat 초기화 완료 (모델: {model})")
        
    def _get_session_history(self, session_id: str) -> ChatMessageHistory:
        """세션 ID로 히스토리 조회 (없으면 새로 생성)"""
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def chat(self, user_input: str, session_id: str = "default") -> str:
        """
        사용자 입력을 받아 AI 응답 반환

        LangChain이 자동으로 처리하는 것들:
        1. 히스토리를 프롬프트에 주입
        2. LLM 호출
        3. 응답 텍스트 파싱

        session_id 활용 예시:
        chat("안녕", session_id="user_001")
        chat("안녕", session_id="user_002")
        → 두 사용자의 대화가 완전히 독립적으로 관리됨

        Args:
            user_input: 사용자 입력 텍스트

        Returns:
            AI 응답 텍스트
        """

        # Chain 실행
        response = self.chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": self.session_id}}
            )

        return response

    def stream_chat(self, user_input: str, session_id: str = "default"):
        """
        스트리밍 응답 생성기

        실무 포인트:
        LangChain의 stream()은 청크를 자동으로 yield한다.
        직접 구현했던 _handle_stream()과 동일한 역할.
        """

        for chunk in self.chain.stream(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        ):
            yield chunk

    def clear_memory(self, session_id: str = "default") -> None:
        """대화 히스토리 초기화"""
        if session_id in self.store:
            self.store[session_id].clear()
        print(f"🗑️  세션 '{session_id}' 히스토리 초기화 완료")

    def get_history(self, session_id: str = "default") -> list:
        """현재 대화 히스토리 반환"""
        return self._get_session_history(session_id).messages
