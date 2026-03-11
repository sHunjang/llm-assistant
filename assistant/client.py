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
    
    2단계 변경사항:
    - chat() 메서드에 tools 파라미터 추가
    - Function Calling 응답 감지 로직 추가
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
        stream: bool = False,
        tools: list[dict] | None = None,    # 2단계에서 추가된 파라미터
    ) -> any:
        """
        Gemini에 메시지를 보내고 응답을 받는 핵심 메서드

        신버전 변경 사항:
        - system_prompt를 config에 직접 전달 (모델 생성 시 주입 방식 → 호출 시 주입 방식)
        - types.GenerateContentConfig 사용
        
        2단계 추가사항:
        tools 파라미터로 사용 가능한 도구 목록을 전달하면
        Gemini가 필요할 때 도구 호출을 요청한다.

        Args:
            messages     : 대화 히스토리
                           [{"role": "user"|"model", "parts": "내용"}, ...]
            system_prompt: 시스템 프롬프트 (매 요청마다 함께 전달)
            temperature  : 창의성 조절 (0.0 ~ 1.0)
            max_tokens   : 최대 응답 토큰 수
            stream       : True면 스트리밍, False면 일반 응답
            tools        : 사용 가능한 도구 스팩 목록
                           None 이면 일반 대화 모드 (1단계와 동일)
                           전달하면 Function Calling 모드
        """

        # tools가 잇으면 Gemini Function Declaration 형식으로 변환
        gemini_tools = None
        if tools:
            gemini_tools = self._build_gemini_tools(tools)
            
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
            tools=gemini_tools,    # 도구 목록 전달
        )
        
        if stream and not tools:
            # 스트리밍은 Function Calling과 함께 사용 시 복잡해짐
            # tools가 없을 때만 스트리밍 사용
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
    
    def _build_gemini_tools(self, tools: list[dict]) -> list:
        """
        우리 도구 스펙 형식을 Gemini FunctinoDeclaration 형식으로 변환
        
        실무 포인트:
            각 LLM API마다 도구 정의 형식이 다르다.
            이 변환 한수 덕분에 definitions.py는 하나만 유지하면서
            여러 LLM API에 대응할 수 있다.
        """
        
        function_declarations = []
        
        for tool in tools:
            func_decl = types.FunctionDeclaration(
                name=tool["name"],
                description=tool["description"],
                parameters=tool.get("parameters", {})
            )
            function_declarations.append(func_decl)
        
        return [types.Tool(function_declarations=function_declarations)]

    
    def has_tool_call(self, response) -> bool:
        """
        응답에 도구 호출 요청이 있는지 확인
        
        신버전 google-genai 응답 구조:
        response.candidates[0].content.parts 를 순회하면서
        function_call 속성이 있는 part를 찾아야 한다.
        
        실무 포인트:
            Gemini 응답은 두 가지 케이스가 있다.
            1. 일반 텍스트 응답 -> 바로 출력
            2. 도구 호출 요청 -> 도구 실행 후 결과 재전송
            이 메서드로 케이스를 구분한다.
        """
        
        try:
            for part in response.candidates[0].content.parts:
                if part.function_call is not None:
                    return True
            return False
        except (IndexError, AttributeError):
            return False
    
    def get_tool_call(self, response) -> tuple[str, dict]:
        """
        응답에서 도구 호출 정보 추출
        
        Returns:
            (tool_name, tool_args) 튜플
        """
        
        for part in response.candidates[0].content.parts:
            if part.function_call is not None:
                tool_name = part.function_call.name
                tool_args = dict(part.function_call.args)
                return tool_name, tool_args

        raise ValueError("응답에 function_call이 없습니다.")


    def _handle_stream(self, response) -> Iterator[str]:
        """스트리밍 응답 처리"""
        for chunk in response:
            if chunk.text:
                yield chunk.text