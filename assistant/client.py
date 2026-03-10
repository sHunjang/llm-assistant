"""
Gemini API 클라이언트 모듈

실무 포인트:
- API 클라이언트는 항상 별도 모듈로 분리한다.
- 이유: 나중에 다른 LLM(GPT, Claude 등)으로 교체할 때
        이 파일만 수정하면 되기 때문이다. (OCP 원칙)
"""

import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from typing import Iterator

# .env 파일에서 환경변수 로드
load_dotenv()


class LLMClient:
    """
    Gemini API 클라이언트 클래스 (google-genai 신버전)

    실무 포인트:
    - 패키지가 바뀌어도 클래스 인터페이스(chat 메서드)는 동일하게 유지
    - main.py나 다른 모듈은 수정할 필요가 없다 → OCP 원칙
    """

    def __init__(self, model: str = "models/gemini-2.5-flash"):
        """
        Args:
            model: 사용할 Gemini 모델명
                   models/gemini-2.5-flash → 최신, 빠르고 무료
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY가 설정되지 않았습니다.\n"
                ".env 파일에 GEMINI_API_KEY=your-key 형식으로 추가해주세요."
            )

        # 신버전 클라이언트 초기화 방식
        self.client = genai.Client(api_key=api_key)
        self.model_name = model

        print(f"✅ LLM 클라이언트 초기화 완료 (모델: {self.model_name})")

    def chat(
        self,
        messages: list[dict],
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False
    ) -> any:
        """
        Gemini에 메시지를 보내고 응답을 받는 핵심 메서드

        신버전 변경 사항:
        - system_prompt를 config에 직접 전달 (모델 생성 시 주입 방식 → 호출 시 주입 방식)
        - types.GenerateContentConfig 사용

        Args:
            messages     : 대화 히스토리
                           [{"role": "user"|"model", "parts": "내용"}, ...]
            system_prompt: 시스템 프롬프트 (매 요청마다 함께 전달)
            temperature  : 창의성 조절 (0.0 ~ 1.0)
            max_tokens   : 최대 응답 토큰 수
            stream       : True면 스트리밍, False면 일반 응답
        """

        # 신버전 설정 방식
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        if stream:
            response = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=messages,
                config=config,
            )
            return self._handle_stream(response)
        else:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=messages,
                config=config,
            )
            return response

    def _handle_stream(self, response) -> Iterator[str]:
        """스트리밍 응답 처리"""
        for chunk in response:
            if chunk.text:
                yield chunk.text