FROM python:3.11-slim
WORKDIR /app
ENV TF_CPP_MIN_LOG_LEVEL=2 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml .
COPY src ./src
RUN pip install --no-deps . && useradd -m -u 10001 app && mkdir /app/state && chown app:app /app/state
COPY --chown=app:app artifacts ./artifacts
USER app
CMD ["uvicorn","mizan.api:app","--host","0.0.0.0","--port","8000","--workers","1"]
