FROM debian:bookworm AS gadget4-build

ARG GADGET4_REF=6fb393b5e25907f2ff06211f67d341c5e40e90d1
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git make g++ python3 pkg-config libopenmpi-dev \
    libhdf5-dev libgsl-dev libfftw3-dev libtirpc-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
RUN git init /opt/gadget4 \
    && cd /opt/gadget4 \
    && git remote add origin https://gitlab.mpcdf.mpg.de/vrs/gadget4.git \
    && git fetch --depth 1 origin "${GADGET4_REF}" \
    && git checkout --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "${GADGET4_REF}"
COPY config/gadget4/Config.sh /opt/gadget4/Config.sh
RUN cd /opt/gadget4 \
    && printf 'SYSTYPE="Generic-gcc"\n' > Makefile.systype \
    && make -j2 LIB_DIR=/usr PYTHON=python3 \
       HDF5_INCL="$(pkg-config --cflags hdf5)" \
       HDF5_LIBS="$(pkg-config --libs hdf5) -lz"

FROM gadget4-build AS rockstar-build
ARG ROCKSTAR_REF=99d56672092e88dbed446f87f6eed87c48ff0e77
RUN git init /opt/rockstar \
    && cd /opt/rockstar \
    && git remote add origin https://github.com/eelregit/rockstar.git \
    && git fetch --depth 1 origin "${ROCKSTAR_REF}" \
    && git checkout --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "${ROCKSTAR_REF}" \
    && make with_hdf5 \
       CFLAGS='-D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64 -D_DEFAULT_SOURCE -D_POSIX_C_SOURCE=200809L -Wall -fno-math-errno -fPIC -fcommon -I/usr/include/tirpc' \
       HDF5_FLAGS="-DH5_USE_16_API -DENABLE_HDF5 $(pkg-config --cflags --libs hdf5) -ltirpc"

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
    libopenmpi3 openmpi-bin libhdf5-103-1 libgsl27 libfftw3-double3 libtirpc3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=gadget4-build /opt/gadget4/Gadget4 /usr/local/bin/Gadget4
COPY --from=gadget4-build /opt/gadget4/Config.sh /opt/gadget4/Config.sh
COPY --from=rockstar-build /opt/rockstar/rockstar /usr/local/bin/rockstar

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
COPY ui ./ui
COPY scripts ./scripts
RUN groupadd --gid 10001 haloforge \
    && useradd --uid 10001 --gid haloforge --create-home haloforge \
    && mkdir -p /var/lib/haloforge/cache /var/lib/haloforge/saved_runs /var/lib/haloforge/exports \
    && chown -R haloforge:haloforge /var/lib/haloforge \
    && python -c "import classy, ui.teaching; print('AxiCLASS binding:', classy.__file__)"
RUN HALOFORGE_DATA_DIR=/opt/haloforge-acceptance \
    python -m scripts.local_gadget_controller_smoke > /opt/HALOFORGE_NBODY_ACCEPTANCE.json

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
