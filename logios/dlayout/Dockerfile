# streamlit/Dockerfile

FROM python:3.10-slim

WORKDIR /src

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    ffmpeg \ 
    libsm6 \
    libxext6 \  
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /src
RUN python -m pip install -r requirements.txt 

COPY /src /src/
WORKDIR  /src
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]