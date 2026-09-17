FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY particlesim ./particlesim
RUN pip install --no-cache-dir .[plot]

ENTRYPOINT ["particlesim"]
CMD ["--help"]
