FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY particlesim ./particlesim
# [serve] brings Panel for `particlesim serve`, the app with runs and live
# modified-theory results (issue #83).
RUN pip install --no-cache-dir .[plot,serve]

# The app, when served:
#   docker run -p 5006:5006 -v "$PWD/runs:/runs" IMAGE \
#     serve --runs /runs --address 0.0.0.0 --allow-websocket-origin localhost:5006
EXPOSE 5006

ENTRYPOINT ["particlesim"]
CMD ["--help"]
