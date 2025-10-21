FROM python:3.10-slim-bookworm

# Install system build tools and libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    python3-dev \
    libfreetype6-dev \
    libpng-dev \
    libssl-dev \
    libffi-dev \
    alsa-utils \
    libasound2 \
    libasound2-dev \
    supervisor \
    portaudio19-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip tools
RUN pip install --upgrade pip setuptools wheel

# Install compatible Cython FIRST so pip doesn't pull a different one in the build env
RUN pip install cython==0.29.36

# Copy and install all other dependencies
COPY requirements.txt .
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

# Copy your app code
COPY . /app

EXPOSE 25
COPY supervisord.conf /etc/supervisor/supervisord.conf
CMD ["supervisord", "-c", "/etc/supervisor/supervisord.conf"]
