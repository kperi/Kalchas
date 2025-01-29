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

RUN pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /src
RUN python -m pip install -r requirements.txt 

COPY /src /src/
WORKDIR  /src
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]