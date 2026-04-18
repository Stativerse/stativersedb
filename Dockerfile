FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=2
ENV OMP_THREAD_LIMIT=2
ENV OPENBLAS_NUM_THREADS=2
ENV MKL_NUM_THREADS=2
ENV NUMEXPR_NUM_THREADS=2
ENV VECLIB_MAXIMUM_THREADS=2
ENV BLIS_NUM_THREADS=2
ENV APP_WORKERS=2

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860} --workers ${APP_WORKERS:-2}"]