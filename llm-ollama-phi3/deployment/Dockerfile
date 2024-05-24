FROM --platform=linux/amd64 ollama/ollama:0.1.38

# Download LLM
RUN ollama serve && ollama pull phi3:3.8b

# Set environment variables to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update the package list, add Deadsnakes PPA, and install Python 3.11
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Set alias for python and pip
RUN alias pip="python3.11 -m pip" \
    alias python="python3.11"

# Set working dir
WORKDIR /app

# Dependencies
COPY ./serve-requirements.txt .
RUN pip install --no-cache-dir  -r serve-requirements.txt

# Copy main script
COPY ./main.py /app/main.py

# Set entrypoint
ENTRYPOINT ["python", "-m", "main"]