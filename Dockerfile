# Headless image for reproducible training / evaluation / GIF recording.
# The interactive Pygame window (`play`) needs a display, so run that natively;
# everything here runs without one via the SDL "dummy" video driver.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    SDL_VIDEODRIVER=dummy \
    SDL_AUDIODRIVER=dummy

# Fonts are needed for the on-screen HUD text even when headless.
RUN apt-get update && apt-get install -y --no-install-recommends \
        fonts-dejavu-core libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the CPU build of PyTorch first to keep the image small and cacheable.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir .

COPY models ./models

ENTRYPOINT ["chaser-runner"]
CMD ["eval", "--episodes", "20"]
