FROM python:3.10-slim

RUN  mkdir -p /app


RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    libxext6 \  
    && rm -rf /var/lib/apt/lists/*

RUN pip install torch==2.0.1 torchvision==0.15.2  --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /app
RUN pip install -r /app/requirements.txt 
 
EXPOSE 8000 8000
RUN mkdir -p /app/ocr_server
COPY  /ocr_server /app/ocr_server
WORKDIR  /app/ocr_server
#CMD ["uvicorn", "srv:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]    
CMD [ "python", "/app/ocr_server/srv.py" ]