FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
  build-essential \
  curl \
  software-properties-common \
  git \
  libsm6 \
  libxext6 \  
  libfontconfig1-dev \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app
RUN pip3 install --no-cache-dir  -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8602/_stcore/health

COPY ./st_app /app/

WORKDIR /app/kalchas

RUN file="$(ls -1 /app)" && echo $file

WORKDIR /app/kalchas/logios

ENV PYTHONPATH=/app/st:$PYTHONPATH
ENTRYPOINT ["streamlit", "run", "/app/st_app/OCR.py", "--server.port=8502",  "--server.baseUrlPath=logios", "--server.address=0.0.0.0"]
