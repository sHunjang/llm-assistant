"""
대화 히스토리 관리 모듈

실무 포인트:
LLM은 기본적으로 무상태(Stateless)다.
이전 대화를 기억하려면 직접 히스토리를 관리해서
매 요청마다 전체 대화 내용을 함께 보내야 한다.

※ Gemini role 규칙
   - 반드시 "user" 와 "model" 이 번갈아 나와야 한다.
   - "user" 로 시작해야 한다.
   - system 프롬프트는 별도로 처리한다.
"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Message:
    """단일 메시지 데이터 클래스"""
    role: str       # "user" 또는 "model"
    content: str    # 메시지 내용
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )


class ConversationManager:
    """
    대화 히스토리 관리 클래스

    핵심 역할:
    1. 대화 히스토리 저장
    2. 컨텍스트 윈도우 초과 방지 (최근 N개만 유지)
    3. Gemini API 형식으로 변환하여 반환
    4. 대화 저장 및 초기화

    실무 팁:
    토큰이 많을수록 비용 증가 + 응답 속도 저하.
    실무에서는 보통 최근 20~30개 대화만 유지하는 전략을 사용한다.
    """

    def __init__(
        self,
        system_prompt: str,
        max_history: int = 20
    ):
        """
        Args:
            system_prompt: AI의 역할과 지시사항 정의
            max_history  : 유지할 최대 메시지 수
                           (user + model 각각 카운트)
        """
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.history: list[Message] = []

        print(f"💬 대화 관리자 초기화 완료 (최대 히스토리: {max_history}개)")

    def add_user_message(self, content: str) -> None:
        """사용자 메시지를 히스토리에 추가"""
        self.history.append(Message(role="user", content=content))

    def add_assistant_message(self, content: str) -> None:
        """
        AI 응답을 히스토리에 추가

        실무 포인트:
        Gemini는 AI 응답의 role이 "model" 이다.
        OpenAI의 "assistant" 와 다르니 주의!
        """
        self.history.append(Message(role="model", content=content))

    def get_messages_for_api(self) -> list[dict]:
        """
        Gemini API에 전송할 형식으로 메시지 목록 반환

        Gemini API 형식:
        [
            {"role": "user",  "parts": "안녕하세요"},
            {"role": "model", "parts": "안녕하세요! 무엇을 도와드릴까요?"},
            {"role": "user",  "parts": "RAG가 뭐야?"},
        ]

        실무 포인트:
        - system 프롬프트는 contents에 포함하지 않는다.
          → LLMClient의 GenerativeModel 생성 시 system_instruction으로 전달
        - 최근 N개만 포함해서 토큰을 절약한다.
        """

        # 최근 N개만 유지
        recent_history = self.history[-self.max_history:]

        # Gemini API 형식으로 변환
        messages = []
        for msg in recent_history:
            messages.append({
                "role": msg.role,
                "parts": msg.content
            })

        return messages

    def get_system_prompt(self) -> str:
        """시스템 프롬프트 반환"""
        return self.system_prompt

    def clear(self) -> None:
        """대화 히스토리 초기화"""
        self.history = []
        print("🗑️  대화 히스토리가 초기화되었습니다.")

    def get_history_count(self) -> int:
        """현재 저장된 메시지 수 반환"""
        return len(self.history)

    def save_to_file(self, filename: str = None) -> str:
        """
        대화 히스토리를 텍스트 파일로 저장

        실무 포인트:
        실제 서비스에서는 DB(PostgreSQL, MongoDB 등)에 저장하지만
        지금은 학습 목적으로 파일로 저장한다.
        """
        if filename is None:
            filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"=== 대화 기록 ({datetime.now().strftime('%Y-%m-%d %H:%M')}) ===\n\n")
            f.write(f"[시스템 프롬프트]\n{self.system_prompt}\n\n")
            f.write("=" * 50 + "\n\n")

            for msg in self.history:
                emoji = "👤" if msg.role == "user" else "🤖"
                f.write(f"{emoji} [{msg.role.upper()}] {msg.timestamp}\n")
                f.write(f"{msg.content}\n\n")

        print(f"💾 대화 저장 완료: {filename}")
        return filename