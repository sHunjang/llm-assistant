"""
Gradio UI — HuggingFace Spaces 배포용

탭 구성:
1. 일반 채팅   → LangChain Chat
2. RAG 채팅    → PDF 업로드 + 질문
3. AI Agent   → 날씨, 계산, 시간 도구

실무 포인트:
Lazy Loading으로 각 탭 첫 사용 시에만 모델 로딩
→ 앱 시작 시간 단축 + 메모리 절약
"""

import os
import gradio as gr
from dotenv import load_dotenv

from langchain_app.chat import LangChainChat
from langchain_app.rag_chain import LangChainRAG
from agent.graph import create_agent_graph
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-2.0-flash-lite")

# ── Lazy Loading 싱글톤 ──────────────────────
_chat_instance = None
_rag_instance = None
_agent_app = None
agent_history: list = []


def get_chat() -> LangChainChat:
    global _chat_instance
    if _chat_instance is None:
        _chat_instance = LangChainChat(model=DEFAULT_MODEL)
    return _chat_instance


def get_rag() -> LangChainRAG:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = LangChainRAG(model=DEFAULT_MODEL)
    return _rag_instance


def get_agent():
    global _agent_app
    if _agent_app is None:
        _agent_app = create_agent_graph(model=DEFAULT_MODEL)
    return _agent_app


# ══════════════════════════════════════════
# 탭 1 — 일반 채팅
# ══════════════════════════════════════════

def chat_respond(message: str, history: list) -> tuple[str, list]:
    if not message.strip():
        return "", history

    try:
        response = get_chat().chat(
            user_input=message,
            session_id="gradio_chat"
        )
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
    except Exception as e:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"❌ 오류: {str(e)}"})

    return "", history


def chat_clear() -> tuple[list, list]:
    if _chat_instance is not None:
        get_chat().clear_memory("gradio_chat")
    return [], []


# ══════════════════════════════════════════
# 탭 2 — RAG 채팅
# ══════════════════════════════════════════

def rag_index(file) -> str:
    if file is None:
        return "❌ PDF 파일을 업로드해주세요."

    try:
        chunks = get_rag().index_document(file.name)
        return f"✅ 인덱싱 완료! {chunks}개 청크 저장됨"
    except Exception as e:
        return f"❌ 인덱싱 실패: {str(e)}"


def rag_respond(question: str, history: list) -> tuple[str, list]:
    if not question.strip():
        return "", history

    if get_rag().chain is None:
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": "❌ PDF를 먼저 업로드해주세요."})
        return "", history

    try:
        answer = get_rag().ask(question)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
    except Exception as e:
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": f"❌ 오류: {str(e)}"})

    return "", history


# ══════════════════════════════════════════
# 탭 3 — AI Agent
# ══════════════════════════════════════════

def agent_respond(message: str, history: list) -> tuple[str, list]:
    global agent_history

    if not message.strip():
        return "", history

    agent_history.append(HumanMessage(content=message))

    try:
        final_response = None
        for state in get_agent().stream(
            {"messages": agent_history},
            stream_mode="values"
        ):
            final_response = state

        if not final_response:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": "❌ Agent 응답 없음"})
            return "", history

        new_messages = final_response["messages"][len(agent_history):]
        final_answer = ""
        tools_used = []

        for msg in new_messages:
            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        tools_used.append(tc["name"])
                elif msg.content:
                    final_answer = msg.content

        if tools_used:
            tools_str = ", ".join(tools_used)
            display = f"🔧 사용 도구: {tools_str}\n\n{final_answer}"
        else:
            display = final_answer

        last_msg = final_response["messages"][-1]
        if isinstance(last_msg, AIMessage) and last_msg.content:
            agent_history.append(last_msg)

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": display})

    except Exception as e:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"❌ 오류: {str(e)}"})

    return "", history


def agent_clear() -> tuple[list, list]:
    global agent_history
    agent_history = []
    return [], []


# ══════════════════════════════════════════
# Gradio UI 구성
# ══════════════════════════════════════════

with gr.Blocks(title="LLM Assistant") as demo:

    gr.Markdown("# 🤖 LLM Assistant")
    gr.Markdown("Gemini 기반 LLM 서비스 | Chat · RAG · Agent")

    with gr.Tabs():

        # ── 탭 1: 일반 채팅 ──────────────────
        with gr.Tab("💬 일반 채팅"):
            gr.Markdown("### LangChain 기반 멀티턴 채팅")

            chat_history = gr.Chatbot(
                height=450,
                label="대화",
                type="messages"
            )
            chat_input = gr.Textbox(
                placeholder="메시지를 입력하세요...",
                label="입력",
                lines=1
            )
            with gr.Row():
                chat_send = gr.Button("전송", variant="primary")
                chat_clear_btn = gr.Button("초기화")

            chat_send.click(
                chat_respond,
                inputs=[chat_input, chat_history],
                outputs=[chat_input, chat_history]
            )
            chat_input.submit(
                chat_respond,
                inputs=[chat_input, chat_history],
                outputs=[chat_input, chat_history]
            )
            chat_clear_btn.click(
                chat_clear,
                outputs=[chat_input, chat_history]
            )

        # ── 탭 2: RAG 채팅 ───────────────────
        with gr.Tab("📄 RAG 채팅"):
            gr.Markdown("### PDF 문서 기반 질의응답")

            with gr.Row():
                with gr.Column(scale=1):
                    rag_file = gr.File(
                        label="PDF 업로드",
                        file_types=[".pdf"]
                    )
                    rag_index_btn = gr.Button("📥 인덱싱", variant="primary")
                    rag_status = gr.Textbox(
                        label="상태",
                        interactive=False,
                        lines=1
                    )

                with gr.Column(scale=2):
                    rag_history = gr.Chatbot(
                        height=400,
                        label="대화",
                        type="messages"
                    )
                    rag_input = gr.Textbox(
                        placeholder="문서에 대해 질문하세요...",
                        label="질문",
                        lines=1
                    )
                    rag_send = gr.Button("질문", variant="primary")

            rag_index_btn.click(
                rag_index,
                inputs=[rag_file],
                outputs=[rag_status]
            )
            rag_send.click(
                rag_respond,
                inputs=[rag_input, rag_history],
                outputs=[rag_input, rag_history]
            )
            rag_input.submit(
                rag_respond,
                inputs=[rag_input, rag_history],
                outputs=[rag_input, rag_history]
            )

        # ── 탭 3: AI Agent ───────────────────
        with gr.Tab("🤖 AI Agent"):
            gr.Markdown("### ReAct Agent — 날씨 · 계산 · 시간 · 검색")
            gr.Markdown(
                "**사용 가능한 도구:** "
                "`날씨 조회` `수학 계산` `현재 시간` `지식 검색`"
            )

            agent_chatbot = gr.Chatbot(
                height=450,
                label="Agent 대화",
                type="messages"
            )
            agent_input = gr.Textbox(
                placeholder="예: 서울 날씨 알려주고 1234*5678 계산해줘",
                label="입력",
                lines=1
            )
            with gr.Row():
                agent_send = gr.Button("실행", variant="primary")
                agent_clear_btn = gr.Button("초기화")

            gr.Examples(
                examples=[
                    ["서울 날씨 알려줘"],
                    ["1234 곱하기 5678은?"],
                    ["지금 몇 시야?"],
                    ["서울이랑 부산 날씨 비교해줘"],
                    ["LangGraph가 뭐야?"],
                ],
                inputs=agent_input
            )

            agent_send.click(
                agent_respond,
                inputs=[agent_input, agent_chatbot],
                outputs=[agent_input, agent_chatbot]
            )
            agent_input.submit(
                agent_respond,
                inputs=[agent_input, agent_chatbot],
                outputs=[agent_input, agent_chatbot]
            )
            agent_clear_btn.click(
                agent_clear,
                outputs=[agent_input, agent_chatbot]
            )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        # share=True: 외부 공유 링크 생성
        theme=gr.themes.Soft()
        # HuggingFace Spaces 배포 시 False
    )