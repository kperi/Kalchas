FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    libxext6 \  
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app
RUN pip install -r requirements.txt 

RUN mkdir /src

COPY  /app/ /src
WORKDIR  /src
CMD ["uvicorn", "ocr_server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]