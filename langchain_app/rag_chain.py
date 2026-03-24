"""
LangChain 기반 RAG Chain 모듈

실무 포인트:
3단계에서 직접 구현한 RAG 파이프라인과 비교.

3단계 직접 구현:
  document_loader.py  → 100줄
  chunker.py          → 100줄
  embedder.py         → 80줄
  vector_store.py     → 100줄
  retriever.py        → 100줄
  합계: 약 480줄

LangChain RAG:
  rag_chain.py        → 약 150줄
  → 코드량 70% 감소!

왜 직접 구현도 배웠냐?
→ 내부 동작 원리를 모르면
  문제가 생겼을 때 디버깅이 불가능하다.
→ LangChain이 추상화해주는 게 뭔지 알아야
  커스터마이징이 가능하다.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()


class LangChainRAG:
    """
    LangChain 기반 RAG 파이프라인 클래스

    3단계 직접 구현 vs LangChain 비교:
    ─────────────────────────────────────────
    DocumentLoader    → PyPDFLoader
    TextChunker       → RecursiveCharacterTextSplitter
    Embedder          → GoogleGenerativeAIEmbeddings
    VectorStore       → Chroma
    RAGRetriever      → vectorstore.as_retriever()

    실무 포인트:
    RecursiveCharacterTextSplitter가
    우리가 직접 만든 TextChunker보다 더 스마트하다.
    문단 → 문장 → 단어 순서로 재귀적으로 분할해서
    의미 단위가 끊기는 걸 최소화한다.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 3,
        persist_directory: str = "./chroma_langchain_db"
    ):
        self.top_k = top_k
        self.persist_directory = persist_directory

        # ── 1. LLM 초기화 ────────────────────────
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.3,    # RAG는 낮은 temperature 권장
        )

        # ── 2. 임베딩 모델 ───────────────────────
        # 실무 포인트:
        # GoogleGenerativeAIEmbeddings 사용 시
        # 별도 모델 다운로드 없이 API로 임베딩 가능
        # 단, API 호출 비용 발생 (무료 티어 범위 내)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask"
        )

        # ── 3. 텍스트 분할기 ─────────────────────
        # RecursiveCharacterTextSplitter:
        # ["\n\n", "\n", " ", ""] 순서로 재귀적 분할
        # → 문단 경계를 최대한 유지
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        # ── 4. RAG 프롬프트 템플릿 ───────────────
        self.prompt = ChatPromptTemplate.from_template("""
다음 문서 내용을 참고해서 질문에 답변해주세요.

규칙:
1. 반드시 제공된 문서 내용만을 근거로 답변하세요.
2. 문서에 없는 내용은 "제공된 문서에서 찾을 수 없습니다"라고 하세요.
3. 답변 마지막에 출처를 명시하세요.

=== 참고 문서 ===
{context}

=== 질문 ===
{question}
""")

        self.vectorstore = None
        self.chain = None

        print(f"✅ LangChain RAG 초기화 완료 (모델: {model})")

    def index_document(self, file_path: str) -> int:
        """
        PDF 문서를 처리하여 Vector DB에 저장

        LangChain 방식 vs 직접 구현 비교:
        ─────────────────────────────────────
        직접 구현:
          loader = DocumentLoader()
          docs = loader.load_pdf(file_path)      # 100줄짜리 클래스
          chunks = chunker.chunk_documents(docs)  # 100줄짜리 클래스
          embedded = embedder.embed_chunks(chunks)# 80줄짜리 클래스
          vector_store.add_documents(embedded)    # 100줄짜리 클래스

        LangChain:
          loader = PyPDFLoader(file_path)
          docs = loader.load_and_split(splitter)  # 끝!
        """
        
        from pathlib import Path
        import hashlib
        
        # ── 파일 해시로 중복 체크 ────────────────
        # 같은 파일이면 재인덱싱 없이 기존 DB 재사용
        file_hash = hashlib.md5(
            Path(file_path).read_bytes()
        ).hexdigest()[:8]
        
        collection_name = f"rag_{file_path}"
        
        # 기존 Vector DB 재사용 시도
        try:
            existing = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
            count = existing.get()["ids"]
            
            if count:
                print(f"✅ 기존 인덱싱 재사용 ({len(count)}개 청크)")
                self.vectorstore = existing
                self._build_chain()
                return len(count)
        except Exception:
            pass

        # 새로 인덱싱
        print(f"{'='*50}")
        print(f"📥 문서 인덱싱 시작: {file_path}")
        print(f"{'='*50}\n")

        print("📄 PDF 로딩 및 청킹 중...")
        loader = PyPDFLoader(file_path)
        documents = loader.load_and_split(self.text_splitter)
        print(f"   ✅ {len(documents)}개 청크 생성\n")

        print("🧠 임베딩 및 Vector DB 저장 중...")
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory=self.persist_directory
        )
        print(f"   ✅ Vector DB 저장 완료\n")

        self._build_chain()
        print(f"✅ 인덱싱 완료! {len(documents)}개 청크 저장\n")
        return len(documents)
        

    def _build_chain(self) -> None:
        """
        LCEL RAG Chain 구성

        LCEL 파이프라인:
        retriever → prompt → llm → output_parser

        실무 포인트:
        retriever는 질문을 받아서 관련 문서를 자동 검색한다.
        vectorstore.as_retriever()로 간단히 생성 가능.
        """

        # Retriever 생성
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )

        # 검색된 문서를 하나의 문자열로 합치는 함수
        def format_docs(docs):
            return "\n\n".join([
                f"[출처: {doc.metadata.get('source', '알 수 없음')} "
                f"{doc.metadata.get('page', '?')+1}페이지]\n{doc.page_content}"
                for doc in docs
            ])

        # LCEL Chain 조립
        # RunnablePassthrough: 입력을 그대로 통과시킴
        self.chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        print("🔗 RAG Chain 구성 완료")

    def ask(self, question: str) -> str:
        """
        질문에 대한 답변 생성

        Args:
            question: 사용자 질문

        Returns:
            문서 기반 AI 답변
        """

        if not self.chain:
            raise RuntimeError(
                "문서가 인덱싱되지 않았습니다. "
                "index_document()를 먼저 실행하세요."
            )

        return self.chain.invoke(question)

    def stream_ask(self, question: str):
        """스트리밍 방식으로 답변 생성"""

        if not self.chain:
            raise RuntimeError("먼저 index_document()를 실행하세요.")

        for chunk in self.chain.stream(question):
            yield chunk