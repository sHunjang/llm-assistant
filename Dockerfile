# Python 3.10 슬림 이미지 사용
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 설치
# (sentence-transformers가 필요로 함)
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 의존성 먼저 복사 (레이어 캐싱 활용)
# requirements.txt가 바뀌지 않으면
# pip install을 다시 실행하지 않음
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스코드 복사
COPY . .

# 로그 디렉토리 생성
RUN mkdir -p logs data

# 포트 노출
EXPOSE 8000

# 서버 실행
# --host 0.0.0.0: 컨테이너 외부에서 접근 가능
# --workers 1: 단일 워커 (개발용)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]