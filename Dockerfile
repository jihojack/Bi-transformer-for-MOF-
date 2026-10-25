FROM python:3.11-slim
WORKDIR /opt/mof
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MOF_MODEL_PATH=/tmp/mof-private/model.onnx
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt
COPY server server
COPY frontend frontend
COPY gunicorn.conf.py .
COPY app/examples app/examples
COPY tools/download_private_model.py tools/download_private_model.py
RUN useradd --create-home --uid 10001 mof
USER mof
EXPOSE 8080
CMD ["sh", "-c", "python -u tools/download_private_model.py && gunicorn --config gunicorn.conf.py --workers 1 --threads 2 --timeout 180 --access-logfile - --error-logfile - --capture-output --log-level info --bind 0.0.0.0:${PORT:-8080} server.app:app"]
