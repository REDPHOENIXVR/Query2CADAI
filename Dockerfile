FROM python:3.11-slim
RUN apt-get update && apt-get install -y git freecad-python3 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python","src/run.py","--mode","single"]