FROM python:3.10-slim

WORKDIR /app
RUN apt-get update && apt-get install -y poppler-utils 

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_APP=run.py
ENV FLASK_DEBUG=true

EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0", "--port=5001", "--reload"]
