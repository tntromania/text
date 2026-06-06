FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        ffmpeg libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m pip install --no-cache-dir --upgrade pip && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python

# PyTorch 2.7 cu CUDA 12.8 → suport nativ Blackwell sm_120
RUN pip install --no-cache-dir \
    torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128

# --no-deps = nu reinstala torch
RUN pip install --no-cache-dir --no-deps simple-lama-inpainting

RUN pip install --no-cache-dir \
    runpod \
    requests \
    opencv-python-headless \
    Pillow \
    huggingface_hub \
    omegaconf \
    kornia \
    easyocr

# Pre-download weights LaMa + EasyOCR în imagine (fără re-download la cold start)
RUN python -c "from simple_lama_inpainting import SimpleLama; SimpleLama(); print('LaMa weights OK')"
RUN python -c "import easyocr; easyocr.Reader(['en', 'ro'], gpu=False, verbose=False); print('EasyOCR models OK')"

COPY handler.py .

CMD ["python", "-u", "handler.py"]
