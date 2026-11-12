FROM python:3.12-slim
RUN pip install --no-cache-dir numpy google-genai
WORKDIR /workspace
CMD ["python", "-u", "trainer/orchestrator.py"]
