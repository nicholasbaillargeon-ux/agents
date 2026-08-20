# Sandbox for model-generated backtest code: no network at run time, read-only
# root filesystem, non-root user, and only the numeric stack installed.
FROM python:3.13-slim

RUN pip install --no-cache-dir "pandas==2.2.3" "numpy==2.1.3" "pyarrow==18.1.0" \
 && useradd --create-home --uid 10001 sandbox

COPY runner.py /app/runner.py
USER sandbox
WORKDIR /app
ENTRYPOINT ["python", "/app/runner.py"]
