FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/AxiCLASS/python \
    OMP_NUM_THREADS=2 \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Kaleido renders the promised PDF/SVG/PNG figures through a local browser.
# Keep Chromium in the runtime image: relying on a user's host browser makes
# Docker exports non-reproducible and previously caused them to fail outright.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gfortran git ca-certificates curl chromium libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir "pip<25" "setuptools<75" wheel "Cython==0.29.36" "numpy==1.26.4"

ARG AXICLASS_REF=1b0a585f86a3dce6babd66e486535368b2799ec7
# AxiCLASS's pyproject imports Cython during metadata generation without
# declaring it as an isolated build dependency. Build the C library first,
# then install the already-pinned local binding with isolation disabled.
RUN git init /opt/AxiCLASS \
    && cd /opt/AxiCLASS \
    && git remote add origin https://github.com/PoulinV/AxiCLASS.git \
    && git fetch --depth 1 origin "${AXICLASS_REF}" \
    && git checkout --detach FETCH_HEAD \
    && git rev-parse HEAD > /opt/AXICLASS_COMMIT \
    && make class libclass.a -j2 \
    && python -m pip install --no-build-isolation .

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY app.py ./
COPY .streamlit ./.streamlit
COPY assets ./assets
COPY config ./config
COPY content ./content
COPY engine ./engine
COPY state ./state
RUN groupadd --gid 10001 haloforge \
    && useradd --uid 10001 --gid haloforge --create-home haloforge \
    && mkdir -p /var/lib/haloforge/cache /var/lib/haloforge/saved_runs /var/lib/haloforge/exports \
    && chown -R haloforge:haloforge /var/lib/haloforge \
    && python -c "import classy; print('AxiCLASS binding:', classy.__file__)"

FROM runtime AS test

COPY requirements-dev.txt ./
RUN python -m pip install --no-cache-dir -r requirements-dev.txt
COPY tests ./tests
COPY docs ./docs
COPY .github ./.github
COPY README.md CHANGELOG.md CITATION.cff CONTRIBUTING.md SECURITY.md \
    Dockerfile docker-compose.yml pyproject.toml .gitignore .dockerignore ./

USER haloforge

FROM runtime AS production

USER haloforge

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=8s --start-period=45s --retries=3 \
  CMD curl --fail http://localhost:7860/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
