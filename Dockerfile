# Stage 1: Builder
FROM python:3.10-slim-bookworm AS builder

WORKDIR /app
COPY requirements_prod.txt .
RUN apt update && pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements_prod.txt

# Stage 2: Runtime (Final Image)
FROM python:3.10-slim-bookworm

# Opecv Dependencies
RUN apt update && apt install -y python3-opencv

WORKDIR /app
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements_prod.txt .
RUN pip install --no-cache --no-index --find-links=/wheels -r requirements_prod.txt && \
    rm -rf /wheels
# Create needed directories
RUN mkdir model; mkdir raw_data; mkdir processed_data

# Copy project last (regularly updated)
COPY baitwatch baitwatch

CMD uvicorn baitwatch.interfaces.api:app --host 0.0.0.0 --port $PORT
