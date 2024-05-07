FROM --platform=linux/amd64 python:3.11.8-slim

WORKDIR /app

# Dependencies
COPY ./serve-requirements.txt .
RUN pip install --no-cache-dir  -r serve-requirements.txt

# Trained model and definition with main script
COPY ./saved_model /app/saved_model
COPY ./main.py /app/main.py

# Set entrypoint
ENTRYPOINT ["python", "-m", "main"]