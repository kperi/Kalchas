# streamlit/Dockerfile

FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    #ffmpeg \ 
    #libsm6 \
    libxext6 \  
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app
RUN python -m pip install -r requirements.txt 