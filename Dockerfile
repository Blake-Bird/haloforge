FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/AxiCLASS/python \
    OMP_NUM_THREADS=2 \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gfortran git ca-certificates curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir "pip<25" "setuptools<75" wheel "Cython==0.29.36" "numpy==1.26.4"

ARG AXICLASS_REF=1b0a585f86a3dce6babd66e486535368b2799ec7
RUN git init /opt/AxiCLASS \
    && cd /opt/AxiCLASS \
    && git remote add origin https://github.com/PoulinV/AxiCLASS.git \
    && git fetch --depth 1 origin "${AXICLASS_REF}" \
    && git checkout --detach FETCH_HEAD \
    && git rev-parse HEAD > /opt/AXICLASS_COMMIT \
    && make PYTHON=python3 -j2

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p data/cache data/saved_runs data/exports \
    && python -c "import classy; print('AxiCLASS binding:', classy.__file__)"

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=8s --start-period=45s --retries=3 \
  CMD curl --fail http://localhost:7860/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
