# Read-only dashboard image. Build from the repo root:
#   docker build -f deploy/Dockerfile -t riskradar-app .
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .
COPY app ./app
COPY data/sample ./data/sample
COPY data/results ./data/results
COPY docs/*.png ./docs/
RUN useradd -m radar && chown -R radar /app
USER radar
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')"
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
