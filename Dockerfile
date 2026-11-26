# ===========================================================================
# Serving image for the fraud model API.
#
# A small, single-stage image is fine here — the dependency tree is modest and
# clarity beats shaving a few MB here. In a real deployment
# you'd use a multi-stage build (build wheels in one stage, copy into a slim
# runtime) and a non-root user; this is called out in docs/learning/08.
# ===========================================================================
FROM python:3.11-slim

# Don't buffer stdout/stderr (so logs appear immediately in `docker logs`), and
# don't write .pyc files inside the container.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies FIRST, using only the requirements file, so this layer is
# cached and only re-runs when requirements change — not on every code edit.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the source and install the package itself (editable-style, but a
# plain install is fine in an image).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .

EXPOSE 8000

# Bind to 0.0.0.0 so the port is reachable from outside the container.
# uvicorn serves the FastAPI app defined in mlops.serving.app:app.
CMD ["uvicorn", "mlops.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
