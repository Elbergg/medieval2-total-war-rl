FROM python:3.12-slim
RUN pip install --no-cache-dir google-genai
WORKDIR /app
COPY server.py .
CMD ["python", "-u", "server.py"]
