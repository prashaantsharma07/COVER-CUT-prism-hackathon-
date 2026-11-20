# Build:  docker build -t prism-track4 .
# Run:    docker run --rm --gpus all -v /videos:/in -v /out:/out prism-track4 /in/P01_11.MP4 /out/P01_11.json
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
RUN chmod +x run.sh
ENTRYPOINT ["/app/run.sh"]
