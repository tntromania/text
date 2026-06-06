FROM node:20-bullseye-slim

# FFmpeg + Python3 + dependențele de build pentru OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    python3 \
    python3-pip \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# opencv + easyocr pentru detecție automată text
RUN pip3 install --no-cache-dir opencv-python-headless easyocr

WORKDIR /app

# Dependențele Node
COPY package*.json ./
RUN npm install

# Codul aplicației (include inpaint_worker.py)
COPY . .

# Foldere necesare
RUN mkdir -p downloads && chmod 777 downloads
RUN mkdir -p public   && chmod 777 public

EXPOSE 3000

CMD ["node", "server.js"]