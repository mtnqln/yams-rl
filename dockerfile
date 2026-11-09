FROM python:3.13-slim
WORKDIR /yams_rl/

COPY pyproject.toml ./
COPY uv.lock ./
COPY README.md ./

RUN pip install --no-cache-dir uv

RUN uv sync --frozen --no-install-project
COPY src/ ./src
RUN uv sync --frozen

ENTRYPOINT ["uv","run","-m","yams_rl.rl.training"]
