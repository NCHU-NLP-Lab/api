FROM tensorflow/tensorflow:2.3.1-gpu

COPY . /app

WORKDIR /app
EXPOSE 8000

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    rsyslog \
    vim \
    wget \
    && rm -rf /var/lib/apt/lists

RUN pip install -r requirements.txt

ENTRYPOINT uvicorn server:app --host 0.0.0.0 --port 8000
