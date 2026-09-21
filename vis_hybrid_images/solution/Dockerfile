FROM python:3.13.2-alpine3.21@sha256:323a717dc4a010fee21e3f1aac738ee10bb485de4e7593ce242b36ee48d6b352
WORKDIR /app
RUN apk add --no-cache \
    build-base \
    bash \
    cmake \
    ninja \
    linux-headers \
    libmagic \
    libjpeg-turbo-dev \
    zlib-dev \
    libxml2-dev \
    libxslt-dev \
    ffmpeg-dev \
    gstreamer-dev \
    && python3 -m ensurepip --upgrade \
    && pip install --no-cache --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["sh"]