FROM python:3.10-slim

WORKDIR /app
RUN apt-get update && apt-get install -y poppler-utils 

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_APP=run.py
ENV FLASK_DEBUG=true
ENV FLASK_ENV=development
ENV PYTHONUNBUFFERED=1

EXPOSE 5001

# Use python run.py instead of flask run for better control
CMD ["python", "run.py"]
