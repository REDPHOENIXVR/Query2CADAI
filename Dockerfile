FROM python:3.11-slim
RUN apt-get update && \
    apt-get install -y git freecad-python3 scrot python3-tk && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

# default to launching the web UI so no pyautogui / GUI automation is required
CMD ["python", "-m", "src.web_ui"]