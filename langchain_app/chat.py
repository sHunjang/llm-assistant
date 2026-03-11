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
  - ConversationBufferMemory가 히스토리 관리
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()


class LangChainChat:
    """
    LangChain 기반 기본 채팅 클래스

    핵심 컴포넌트:
    1. ChatGoogleGenerativeAI → LLM 모델
    2. ChatPromptTemplate     → 프롬프트 템플릿
    3. ConversationBufferMemory → 대화 히스토리
    4. StrOutputParser        → 응답 파싱
    5. LCEL Chain (|)         → 전체 파이프라인 연결
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
        self.llm = ChatGoogleGenerativeAI(
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

        # ── 3. 메모리 ────────────────────────────
        # 직접 구현: ConversationManager 클래스 직접 만들었음
        # LangChain: ConversationBufferMemory 가 자동 관리
        # return_messages=True: 메시지 객체 형태로 반환
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )

        # ── 4. 출력 파서 ─────────────────────────
        # 직접 구현: response.candidates[0].content.parts[0].text
        # LangChain: StrOutputParser() 가 자동으로 텍스트 추출
        self.output_parser = StrOutputParser()

        # ── 5. LCEL Chain 조립 ───────────────────
        # | 연산자로 컴포넌트를 파이프라인처럼 연결
        # 실행 순서: prompt → llm → output_parser
        self.chain = (
            RunnablePassthrough.assign(
                chat_history=lambda x: self.memory.load_memory_variables({})["chat_history"]
            )
            | self.prompt
            | self.llm
            | self.output_parser
        )

        print(f"✅ LangChain Chat 초기화 완료 (모델: {model})")

    def chat(self, user_input: str) -> str:
        """
        사용자 입력을 받아 AI 응답 반환

        LangChain이 자동으로 처리하는 것들:
        1. 히스토리를 프롬프트에 주입
        2. LLM 호출
        3. 응답 텍스트 파싱

        Args:
            user_input: 사용자 입력 텍스트

        Returns:
            AI 응답 텍스트
        """

        # Chain 실행
        response = self.chain.invoke({"input": user_input})

        # 메모리에 대화 저장
        # 직접 구현: conversation.add_user_message() / add_assistant_message()
        # LangChain: save_context()로 한 번에 저장
        self.memory.save_context(
            {"input": user_input},
            {"output": response}
        )

        return response

    def stream_chat(self, user_input: str):
        """
        스트리밍 응답 생성기

        실무 포인트:
        LangChain의 stream()은 청크를 자동으로 yield한다.
        직접 구현했던 _handle_stream()과 동일한 역할.
        """

        # 히스토리 로드
        chat_history = self.memory.load_memory_variables({})["chat_history"]

        # 스트리밍 실행
        full_response = ""
        for chunk in self.chain.stream({
            "input": user_input,
            "chat_history": chat_history
        }):
            full_response += chunk
            yield chunk

        # 메모리 저장
        self.memory.save_context(
            {"input": user_input},
            {"output": full_response}
        )

    def clear_memory(self) -> None:
        """대화 히스토리 초기화"""
        self.memory.clear()
        print("🗑️  대화 히스토리 초기화 완료")

    def get_history(self) -> list:
        """현재 대화 히스토리 반환"""
        return self.memory.load_memory_variables({})["chat_history"]