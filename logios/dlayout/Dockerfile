FROM python:3.10-slim

RUN  mkdir -p /app/dlayout
WORKDIR /app/dlayout

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

COPY requirements.txt /src/dlayout/requirements.txt
RUN pip install -r /src/dlayout/requirements.txt

COPY /src /app/dlayout/src
 

RUN echo "Current path is: $(pwd)"


EXPOSE 8001 8001
EXPOSE 8601 8601
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]