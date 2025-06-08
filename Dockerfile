FROM python:3.12-slim

# 1. 보안 목적의 사용자 생성
RUN adduser --no-create-home --disabled-login appuser

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Cloud SQL Unix socket 연결용 디렉토리
RUN mkdir -p /cloudsql

# 5. VOLUME 설정 (선택적)
VOLUME ["/cloudsql"]

# 6. 파이썬 import 경로
ENV PYTHONPATH=/app

# 7. 애플리케이션 소스 복사
COPY . .

# 8. 비권한 사용자로 전환 (보안 강화)
USER appuser

# 9. 애플리케이션 실행 (PORT는 Cloud Run에서 자동 설정)
CMD sh -c "gunicorn --bind :$PORT --workers 1 --threads 8 main:app"
