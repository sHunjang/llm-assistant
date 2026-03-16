---
title: LLM Assistant
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.23.3
app_file: app.py
pinned: false
---

# LLM Assistant

Gemini 기반 LLM 서비스

## 기능
- 💬 일반 채팅 (LangChain 멀티턴)
- 📄 RAG 채팅 (PDF 업로드 + 질의응답)
- 🤖 AI Agent (날씨, 계산, 시간, 검색 도구)

## 기술 스택
- Gemini API, LangChain, LangGraph
- ChromaDB, sentence-transformers
- FastAPI, Gradio