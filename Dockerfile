FROM tensorflow/tensorflow:2.3.1-gpu

WORKDIR /app
EXPOSE 8000

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    rsyslog \
    vim \
    wget \
    && rm -rf /var/lib/apt/lists

# Install modules
COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre download language models
COPY language_models.py .
RUN python language_models.py

# Copy whole app
COPY . .

ENTRYPOINT uvicorn server:app --host 0.0.0.0 --port 8000
