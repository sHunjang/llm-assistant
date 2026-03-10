"""
Gemini API 클라이언트 모듈

실무 포인트:
- API 클라이언트는 항상 별도 모듈로 분리한다.
- 이유: 나중에 다른 LLM(GPT, Claude 등)으로 교체할 때
        이 파일만 수정하면 되기 때문이다. (OCP 원칙)
"""

import os
from google import generativeai as genai
from dotenv import load_dotenv
from typing import Iterator

# .env 파일에서 환경변수 로드
load_dotenv()


class LLMClient:
    """
    Gemini API 클라이언트 클래스

    실무 포인트:
    - 모델명을 하드코딩하지 않고 파라미터로 받는다.
    - 스트리밍 / 논스트리밍 모두 지원한다.
    - API 키가 없으면 즉시 에러를 발생시킨다. (빠른 실패 원칙)
    """

    def __init__(self, model: str = "gemini-1.5-flash"):
        """
        Args:
            model: 사용할 Gemini 모델명
                   gemini-1.5-flash → 빠르고 무료 (학습/개발용) ✅
                   gemini-1.5-pro   → 고성능 (복잡한 작업용)
        """
        # API 키가 없으면 즉시 에러 발생
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY가 설정되지 않았습니다.\n"
                ".env 파일에 GEMINI_API_KEY=your-key 형식으로 추가해주세요."
            )

        # Gemini API 초기화
        genai.configure(api_key=api_key)
        self.model_name = model

        # GenerativeModel 인스턴스 생성
        # generation_config는 나중에 chat() 호출 시 덮어쓸 수 있다
        self.model = genai.GenerativeModel(model_name=self.model_name)

        print(f"✅ LLM 클라이언트 초기화 완료 (모델: {self.model_name})")

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False
    ) -> any:
        """
        Gemini에 메시지를 보내고 응답을 받는 핵심 메서드

        Args:
            messages : 대화 히스토리
                       [{"role": "user" | "model", "parts": "내용"}, ...]

                       ※ Gemini는 OpenAI와 role 명칭이 다르다!
                          OpenAI : "assistant" → Gemini : "model"
                          OpenAI : "system"    → Gemini : system_instruction (별도 처리)

            temperature: 창의성 조절 (0.0 ~ 1.0)
                         0.0 → 항상 같은 답 (사실 기반 QA에 적합)
                         0.7 → 자연스러운 일반 대화
                         1.0 → 창의적 글쓰기

            max_tokens : 최대 응답 토큰 수 (비용/속도 제어)
            stream     : True면 스트리밍, False면 일반 응답

        Returns:
            stream=False → Gemini response 객체
            stream=True  → 스트리밍 제너레이터
        """

        # 생성 설정
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        if stream:
            response = self.model.generate_content(
                contents=messages,
                generation_config=generation_config,
                stream=True
            )
            return self._handle_stream(response)
        else:
            response = self.model.generate_content(
                contents=messages,
                generation_config=generation_config,
                stream=False
            )
            return response

    def _handle_stream(self, response) -> Iterator[str]:
        """
        스트리밍 응답을 처리하는 내부 메서드

        실무 포인트:
        ChatGPT처럼 글자가 하나씩 출력되는 효과는
        실제로 스트리밍 API를 사용하는 것이다.
        """
        for chunk in response:
            if chunk.text:
                yield chunk.text