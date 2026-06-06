#!/usr/bin/env python3
"""
runpod_handler.py — Caption Eraser AI Tier (RunPod Serverless)

Primește un video + boxes, aplică LaMa inpainting per frame (GPU),
returnează videoul procesat ca base64.

Input JSON:
  {
    "video_url":    "https://...",   # URL public la video (preferat)
    "video_base64": "...",           # alternativ: video encodat base64 (<50MB)
    "boxes": [{"x":px,"y":py,"w":pw,"h":ph}, ...],  # pixeli absoluti
    "width": 1920, "height": 1080, "fps": 30.0
  }

Output JSON:
  { "video_base64": "..." }   # MP4 procesat, encodat base64
"""

import os
import sys
import json
import base64
import tempfile
import subprocess
import runpod
import requests
import numpy as np
import cv2
from PIL import Image

# LaMa se încarcă O SINGURĂ DATĂ la pornirea workerului (nu per job)
print("[INIT] Încărcare model LaMa...", flush=True)
from simple_lama_inpainting import SimpleLama
LAMA = SimpleLama()
print("[INIT] Model LaMa încărcat OK", flush=True)


def auto_detect_boxes(cap, width: int, height: int) -> list:
    """Detectează text în primul frame cu EasyOCR (GPU)."""
    try:
        import easyocr
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame_bgr = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if not ret:
            return []
        reader = easyocr.Reader(['en', 'ro'], gpu=True, verbose=False)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = reader.readtext(frame_rgb, detail=1)
        PAD = 8
        boxes = []
        for (bbox, text, conf) in results:
            if conf < 0.2 or not str(text).strip():
                continue
            pts = np.array(bbox, dtype=np.int32)
            x, y, bw, bh = cv2.boundingRect(pts)
            boxes.append({
                'x': int(max(0, x - PAD)),
                'y': int(max(0, y - PAD)),
                'w': int(min(width - max(0, x - PAD), bw + 2 * PAD)),
                'h': int(min(height - max(0, y - PAD), bh + 2 * PAD)),
            })
        print(f"[AUTO-DETECT] EasyOCR: {len(boxes)} zone găsite → {boxes}", flush=True)
        return boxes
    except Exception as e:
        print(f"[AUTO-DETECT] EasyOCR err: {e}", flush=True)
        return []


def build_mask(boxes: list, width: int, height: int) -> np.ndarray:
    """Construiește masca uint8 (0/255) din lista de boxes cu padding."""
    mask = np.zeros((height, width), dtype=np.uint8)
    PAD = 4
    for b in boxes:
        x1 = max(0, int(b['x']) - PAD)
        y1 = max(0, int(b['y']) - PAD)
        x2 = min(width,  int(b['x']) + int(b['w']) + PAD)
        y2 = min(height, int(b['y']) + int(b['h']) + PAD)
        mask[y1:y2, x1:x2] = 255
    return mask


def process_video(input_path: str, boxes: list, width: int, height: int, fps: float) -> str:
    """
    Procesează fiecare frame cu LaMa inpainting.
    Returnează calea la fișierul MP4 final (cu audio original).
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Nu pot deschide videoul: {input_path}")

    if not boxes:
        print("[PROC] Niciun box furnizat — auto-detectare text...", flush=True)
        boxes = auto_detect_boxes(cap, width, height)
        if not boxes:
            cap.release()
            raise RuntimeError("Auto-detectare eșuată: niciun text găsit. Furnizează boxes manual.")

    mask_np = build_mask(boxes, width, height)
    mask_pil = Image.fromarray(mask_np)

    raw_out = input_path + "_raw.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(raw_out, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # LaMa lucrează cu PIL RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        result_pil = LAMA(frame_pil, mask_pil)

        result_bgr = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
        writer.write(result_bgr)

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"[PROC] {frame_idx} frames procesate...", flush=True)

    cap.release()
    writer.release()
    print(f"[PROC] Total: {frame_idx} frames", flush=True)

    # Re-mux cu audio original (fără re-encodare audio)
    final_out = input_path + "_final.mp4"
    subprocess.run([
        'ffmpeg', '-y', '-nostats', '-loglevel', 'error',
        '-i', raw_out,
        '-i', input_path,
        '-map', '0:v:0',
        '-map', '1:a?',
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
        '-c:a', 'copy',
        '-movflags', '+faststart',
        final_out,
    ], check=True)

    os.remove(raw_out)
    return final_out


def handler(job):
    """RunPod job handler — apelat pentru fiecare job."""
    job_input = job.get('input', {})

    boxes  = job_input.get('boxes',  [])
    width  = int(job_input.get('width',  0))
    height = int(job_input.get('height', 0))
    fps    = float(job_input.get('fps',  30.0))

    if width == 0 or height == 0:
        return {'error': 'Input invalid: width/height lipsesc'}

    # Descărcăm / decodăm videoul
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        input_path = tmp.name

    try:
        if 'video_url' in job_input:
            print(f"[DL] Descărcare video de la: {job_input['video_url']}", flush=True)
            resp = requests.get(job_input['video_url'], timeout=300, stream=True)
            resp.raise_for_status()
            with open(input_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    f.write(chunk)
            print(f"[DL] Descărcat: {os.path.getsize(input_path) / 1024 / 1024:.1f} MB", flush=True)

        elif 'video_base64' in job_input:
            print("[DL] Decodare video din base64...", flush=True)
            with open(input_path, 'wb') as f:
                f.write(base64.b64decode(job_input['video_base64']))
            print(f"[DL] Decodat: {os.path.getsize(input_path) / 1024 / 1024:.1f} MB", flush=True)

        else:
            return {'error': 'Niciun video furnizat (video_url sau video_base64)'}

        # Procesăm
        output_path = process_video(input_path, boxes, width, height, fps)

        # Encodăm output ca base64
        with open(output_path, 'rb') as f:
            video_b64 = base64.b64encode(f.read()).decode('utf-8')

        print(f"[DONE] Output: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB → base64 gata", flush=True)
        return {'video_base64': video_b64}

    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        return {'error': str(e)}

    finally:
        for p in [input_path, input_path + '_raw.mp4', input_path + '_final.mp4']:
            try: os.remove(p)
            except: pass


# RunPod pornește workerul și ascultă joburi
runpod.serverless.start({'handler': handler})
