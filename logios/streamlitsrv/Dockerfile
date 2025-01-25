FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    #ffmpeg \ 
    libsm6 \
    libxext6 \  
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app
RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

COPY ./st /app/

WORKDIR /app/kalchas

RUN file="$(ls -1 /app)" && echo $file

WORKDIR /app/st

ENTRYPOINT ["streamlit", "run", "/app/st/OCR.py", "--server.port=8501", "--server.address=0.0.0.0"]
