"""
HuggingFace Spaces 배포 스크립트

실행 순서:
1. Space 생성
2. 필요 파일 업로드
3. Secret 설정 안내
"""

from huggingface_hub import HfApi, create_repo
import os

# ── 설정 ─────────────────────────────────────
HF_TOKEN    = input("HuggingFace Token 입력: ").strip()
HF_USERNAME = input("HuggingFace 사용자명 입력: ").strip()
SPACE_NAME  = "llm-assistant"
REPO_ID     = f"{HF_USERNAME}/{SPACE_NAME}"

# 업로드할 파일 목록
UPLOAD_FILES = [
    # 진입점
    "app.py",
    "requirements.txt",
    "README.md",

    # 핵심 모듈
    "langchain_app/__init__.py",
    "langchain_app/chat.py",
    "langchain_app/rag_chain.py",
    "langchain_app/memory_chat.py",

    "agent/__init__.py",
    "agent/state.py",
    "agent/tools.py",
    "agent/graph.py",

    "core/__init__.py",
    "core/config.py",
    "core/logger.py",
    "core/cache.py",
    "core/exceptions.py",

    "prompts/system_prompts.py",
]

api = HfApi(token=HF_TOKEN)


def create_space():
    """Space 생성"""
    print(f"\n🚀 Space 생성 중: {REPO_ID}")
    try:
        create_repo(
            repo_id=REPO_ID,
            repo_type="space",
            space_sdk="gradio",
            token=HF_TOKEN,
            exist_ok=True,      # 이미 있어도 에러 안 남
            private=False,
        )
        print(f"✅ Space 생성 완료: https://huggingface.co/spaces/{REPO_ID}")
    except Exception as e:
        print(f"❌ Space 생성 실패: {e}")
        raise


def upload_files():
    """파일 업로드"""
    print(f"\n📤 파일 업로드 중...")

    success, fail = [], []

    for file_path in UPLOAD_FILES:
        if not os.path.exists(file_path):
            print(f"   ⚠️  건너뜀 (없음): {file_path}")
            continue

        try:
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file_path,
                repo_id=REPO_ID,
                repo_type="space",
            )
            print(f"   ✅ {file_path}")
            success.append(file_path)
        except Exception as e:
            print(f"   ❌ {file_path}: {e}")
            fail.append(file_path)

    print(f"\n📊 업로드 결과: 성공 {len(success)}개 / 실패 {len(fail)}개")
    return len(fail) == 0


def print_next_steps():
    """다음 단계 안내"""
    print(f"""
{'='*55}
✅ 배포 완료!

📌 다음 단계 — Secret 설정 (필수!)
{'='*55}
1. 아래 URL 접속:
   https://huggingface.co/spaces/{REPO_ID}/settings

2. "Variables and secrets" 섹션에서
   "New secret" 클릭

3. 아래 값 입력:
   Name  : GEMINI_API_KEY
   Value : (실제 Gemini API 키)

4. Save 클릭 → Space 자동 재빌드

📌 배포된 앱 URL:
   https://huggingface.co/spaces/{REPO_ID}
{'='*55}
""")


def main():
    print("🤗 HuggingFace Spaces 배포 시작")
    print(f"   Space: {REPO_ID}\n")

    create_space()
    success = upload_files()

    if success:
        print_next_steps()
    else:
        print("\n⚠️  일부 파일 업로드 실패. 위 목록 확인 후 재시도해줘.")


if __name__ == "__main__":
    main()