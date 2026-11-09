FROM pytorch:3.13-slim
WORKDIR /yams_rl/

COPY pyproject.toml ./
COPY uv.lock ./

RUN pip install --no-cache-dir uv
RUN uv sync

COPY src/ ./src

CMD uv run -m src.yams_rl.rl.training
