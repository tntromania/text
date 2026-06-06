#!/usr/bin/env python3
import os, sys, subprocess, traceback, time

print("[INIT] Python start...", flush=True)

try:
    import base64, tempfile, requests
    import numpy as np, cv2, runpod
    from PIL import Image
    print("[INIT] Importuri OK", flush=True)
except Exception as e:
    print(f"[FATAL] {e}", flush=True); traceback.print_exc(); sys.exit(1)

# ─────────────────────────────────────────────────────────────
# GPU / LaMa init
# ─────────────────────────────────────────────────────────────
try:
    import torch
    USE_CUDA = torch.cuda.is_available()
    if USE_CUDA:
        # PyTorch 2.7 + cu128 → suport nativ Blackwell sm_120
        DEVICE = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        try:
            cap = torch.cuda.get_device_capability(0)
            print(f"[INIT] GPU: {gpu_name} sm_{cap[0]}{cap[1]}", flush=True)
        except Exception:
            print(f"[INIT] GPU: {gpu_name}", flush=True)
        # cudnn benchmark = autotune kernels pentru dimensiunea ROI
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device('cpu')
        print("[INIT] Niciun GPU CUDA detectat → CPU", flush=True)

    from simple_lama_inpainting import SimpleLama
    print("[INIT] Incarcare LaMa...", flush=True)
    LAMA = SimpleLama()
    # Forțăm device-ul efectiv (în caz că simple_lama default-ează altundeva)
    try:
        LAMA.model.to(DEVICE)
        LAMA.device = DEVICE
    except Exception:
        pass
    # fp16 e DEZACTIVAT pentru LaMa: modelul folosește Fourier convolutions (FFC),
    # iar cuFFT nu suportă fp16 decât pe dimensiuni putere-de-2.
    # ROI-urile noastre au dimensiuni arbitrare (ex. 20×64, 21×56) → cuFFT crapă.
    # Rămânem pe fp32 pe GPU — tot e 5-10× mai rapid decât CPU.
    USE_FP16 = False
    if USE_CUDA:
        print("[INIT] LaMa → fp32 pe GPU (fp16 incompatibil cu cuFFT non-pow2)", flush=True)
    print(f"[INIT] LaMa pe {'CUDA' if USE_CUDA else 'CPU'}!", flush=True)
except Exception as e:
    print(f"[FATAL] {e}", flush=True); traceback.print_exc(); sys.exit(1)

# EasyOCR reader — încărcat o singură dată, reutilizat între job-uri
_OCR_READER = None
def get_ocr_reader():
    global _OCR_READER
    if _OCR_READER is None:
        import easyocr
        _OCR_READER = easyocr.Reader(['en', 'ro'], gpu=USE_CUDA, verbose=False)
        print(f"[INIT] EasyOCR ready ({'GPU' if USE_CUDA else 'CPU'})", flush=True)
    return _OCR_READER

# Detectăm dacă NVENC e disponibil (1× per worker)
def _detect_nvenc():
    if not USE_CUDA:
        return False
    try:
        r = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], capture_output=True, text=True, timeout=5)
        return 'h264_nvenc' in r.stdout
    except Exception:
        return False
HAS_NVENC = _detect_nvenc()
print(f"[INIT] Encoder: {'h264_nvenc (GPU)' if HAS_NVENC else 'libx264 (CPU)'}", flush=True)


# ─────────────────────────────────────────────────────────────
# OCR — multi-frame detect (uniunea tuturor box-urilor)
# ─────────────────────────────────────────────────────────────
def auto_detect_boxes(video_path, width, height):
    """Detectează text la 4 momente diferite (10/30/60/90%) și unește box-urile."""
    try:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return []

        reader = get_ocr_reader()
        PAD = 18                   # padding mai generos pentru glow/shadow/stroke
        all_boxes = []
        samples = [0.10, 0.30, 0.60, 0.90]
        for s in samples:
            frame_idx = max(0, min(int(total * s), total - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame_bgr = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            try:
                results = reader.readtext(frame_rgb, detail=1, paragraph=False)
            except Exception as e:
                print(f"[AUTO-DETECT] OCR err @{s:.0%}: {e}", flush=True)
                continue
            for (bbox, text, conf) in results:
                if conf < 0.2 or not str(text).strip():
                    continue
                pts = np.array(bbox, dtype=np.int32)
                x, y, bw, bh = cv2.boundingRect(pts)
                all_boxes.append({
                    'x': int(max(0, x - PAD)),
                    'y': int(max(0, y - PAD)),
                    'w': int(min(width - max(0, x - PAD), bw + 2 * PAD)),
                    'h': int(min(height - max(0, y - PAD), bh + 2 * PAD)),
                })
        cap.release()

        merged = _merge_overlapping(all_boxes, width, height, gap=12)
        print(f"[AUTO-DETECT] {len(all_boxes)} raw → {len(merged)} merged", flush=True)
        return merged
    except Exception as e:
        print(f"[AUTO-DETECT] Err: {e}", flush=True)
        traceback.print_exc()
        return []


def _merge_overlapping(boxes, W, H, gap=8):
    """Unește box-urile care se suprapun sau sunt foarte apropiate."""
    if not boxes:
        return []
    rects = [(b['x'], b['y'], b['x'] + b['w'], b['y'] + b['h']) for b in boxes]
    used = [False] * len(rects)
    out = []
    for i, r in enumerate(rects):
        if used[i]:
            continue
        x1, y1, x2, y2 = r
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j in range(len(rects)):
                if used[j]:
                    continue
                rx1, ry1, rx2, ry2 = rects[j]
                if (rx1 <= x2 + gap and rx2 + gap >= x1 and
                    ry1 <= y2 + gap and ry2 + gap >= y1):
                    x1, y1 = min(x1, rx1), min(y1, ry1)
                    x2, y2 = max(x2, rx2), max(y2, ry2)
                    used[j] = True
                    changed = True
        out.append({
            'x': max(0, x1), 'y': max(0, y1),
            'w': min(W, x2) - max(0, x1),
            'h': min(H, y2) - max(0, y1),
        })
    return out


# ─────────────────────────────────────────────────────────────
# Mask building — dilation gaussian peste fiecare box
# ─────────────────────────────────────────────────────────────
def build_mask_for_boxes(boxes, W, H, feather=6):
    """Mască per-uniune dilatată suplimentar cu gaussian blur (anti-halo)."""
    mask = np.zeros((H, W), dtype=np.uint8)
    DILATE = 6
    for b in boxes:
        x1 = max(0, int(b['x']) - DILATE)
        y1 = max(0, int(b['y']) - DILATE)
        x2 = min(W, int(b['x']) + int(b['w']) + DILATE)
        y2 = min(H, int(b['y']) + int(b['h']) + DILATE)
        mask[y1:y2, x1:x2] = 255
    if feather > 0:
        k = feather * 2 + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        mask[mask > 0] = 255
    return mask


# ─────────────────────────────────────────────────────────────
# ROI clustering — grupăm box-urile apropiate într-un singur ROI,
# dar păstrăm separate cele depărtate (evităm un ROI uriaș degeaba).
# ─────────────────────────────────────────────────────────────
def cluster_boxes_into_rois(boxes, W, H, context=40, max_gap=120):
    """Returnează listă de (rx1,ry1,rx2,ry2) — câte un ROI per cluster."""
    if not boxes:
        return []
    clusters = _merge_overlapping(boxes, W, H, gap=max_gap)
    rois = []
    for c in clusters:
        rx1 = max(0, c['x'] - context)
        ry1 = max(0, c['y'] - context)
        rx2 = min(W, c['x'] + c['w'] + context)
        ry2 = min(H, c['y'] + c['h'] + context)
        rois.append((rx1, ry1, rx2, ry2))
    return rois


# ─────────────────────────────────────────────────────────────
# Inpainting pe un ROI — fp16 dacă suntem pe GPU
# ─────────────────────────────────────────────────────────────
@torch.inference_mode()
def inpaint_roi(roi_bgr_np, roi_mask_pil):
    roi_rgb_pil = Image.fromarray(cv2.cvtColor(roi_bgr_np, cv2.COLOR_BGR2RGB))
    if USE_FP16:
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            return LAMA(roi_rgb_pil, roi_mask_pil)
    return LAMA(roi_rgb_pil, roi_mask_pil)


# ─────────────────────────────────────────────────────────────
# Main process
# ─────────────────────────────────────────────────────────────
def process_video(input_path, boxes, width, height, fps):
    import json as _json
    probe = subprocess.run(
        ['ffprobe','-v','error','-select_streams','v:0',
         '-show_entries','stream=width,height,r_frame_rate','-of','json', input_path],
        capture_output=True, text=True, timeout=30
    )
    if probe.returncode != 0:
        raise RuntimeError(f"ffprobe: {probe.stderr[:200]}")
    meta = _json.loads(probe.stdout)['streams'][0]
    actual_w, actual_h = int(meta['width']), int(meta['height'])
    num, den = map(int, meta['r_frame_rate'].split('/'))
    actual_fps = num / den
    print(f"[PROC] {actual_w}x{actual_h} @ {actual_fps:.4f}fps", flush=True)

    if not boxes:
        print("[PROC] Boxes goale — auto-detectare text...", flush=True)
        boxes = auto_detect_boxes(input_path, actual_w, actual_h)
        if not boxes:
            raise RuntimeError("Auto-detectare eșuată: niciun text. Furnizează boxes manual.")
        print(f"[PROC] Auto-detect: {len(boxes)} zone", flush=True)

    # Mască full-frame (pentru lookup) + clustere ROI
    mask_np = build_mask_for_boxes(boxes, actual_w, actual_h, feather=6)
    mask_full_pil = Image.fromarray(mask_np)
    rois = cluster_boxes_into_rois(boxes, actual_w, actual_h, context=40, max_gap=120)
    if not rois:
        raise RuntimeError("Niciun ROI valid după clustering.")

    # Pre-crop mask per ROI (o singură dată)
    roi_data = []
    total_roi_px = 0
    for (rx1, ry1, rx2, ry2) in rois:
        rw, rh = rx2 - rx1, ry2 - ry1
        roi_mask = mask_full_pil.crop((rx1, ry1, rx2, ry2))
        roi_data.append({'box': (rx1, ry1, rx2, ry2), 'mask': roi_mask, 'wh': (rw, rh)})
        total_roi_px += rw * rh
    speedup = round((actual_w * actual_h) / max(1, total_roi_px))
    print(f"[PROC] {len(rois)} ROI(s), total {total_roi_px}px — speedup ~{speedup}x", flush=True)
    for i, rd in enumerate(roi_data):
        rx1, ry1, rx2, ry2 = rd['box']
        print(f"[PROC]   ROI #{i+1}: {rx2-rx1}x{ry2-ry1} @ ({rx1},{ry1})", flush=True)

    frame_bytes = actual_w * actual_h * 3
    final = input_path + "_final.mp4"

    # Decoder
    dec = subprocess.Popen([
        'ffmpeg','-y','-loglevel','error',
        '-i', input_path,
        '-map','0:v:0',
        '-vf', f'scale={actual_w}:{actual_h},format=bgr24',
        '-f','rawvideo','pipe:1'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Encoder — NVENC pe GPU, altfel x264
    if HAS_NVENC:
        enc_args = [
            'ffmpeg','-y','-loglevel','error',
            '-f','rawvideo','-pixel_format','bgr24',
            '-video_size', f'{actual_w}x{actual_h}',
            '-framerate', str(actual_fps),
            '-i','pipe:0','-i', input_path,
            '-map','0:v:0','-map','1:a?',
            '-c:v','h264_nvenc','-preset','p4','-tune','hq',
            '-rc','vbr','-cq','23','-b:v','0',
            '-pix_fmt','yuv420p',
            '-c:a','copy','-movflags','+faststart', final,
        ]
    else:
        enc_args = [
            'ffmpeg','-y','-loglevel','error',
            '-f','rawvideo','-pixel_format','bgr24',
            '-video_size', f'{actual_w}x{actual_h}',
            '-framerate', str(actual_fps),
            '-i','pipe:0','-i', input_path,
            '-map','0:v:0','-map','1:a?',
            '-c:v','libx264','-preset','fast','-crf','23','-pix_fmt','yuv420p',
            '-c:a','copy','-movflags','+faststart', final,
        ]
    enc = subprocess.Popen(enc_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    n = 0
    write_err = None
    t_start = time.time()
    try:
        while True:
            raw = dec.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            bgr_in = np.frombuffer(raw, dtype=np.uint8).reshape((actual_h, actual_w, 3)).copy()

            # Procesăm fiecare ROI separat — modificăm in-place
            for rd in roi_data:
                rx1, ry1, rx2, ry2 = rd['box']
                roi_w, roi_h = rd['wh']
                roi_bgr = bgr_in[ry1:ry2, rx1:rx2].copy()
                result_pil = inpaint_roi(roi_bgr, rd['mask'])
                # Redimensionăm doar dacă LaMa a schimbat shape-ul (rar dar posibil)
                if result_pil.size != (roi_w, roi_h):
                    result_pil = result_pil.resize((roi_w, roi_h), Image.LANCZOS)
                result_bgr = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
                bgr_in[ry1:ry2, rx1:rx2] = result_bgr

            if n == 0:
                print(f"[DBG] frame={bgr_in.shape} rois={len(roi_data)}", flush=True)

            try:
                enc.stdin.write(bgr_in.tobytes())
            except (BrokenPipeError, ValueError, OSError) as e:
                write_err = e; break
            n += 1
            if n % 30 == 0:
                dt = time.time() - t_start
                fps_now = n / dt if dt > 0 else 0
                print(f"[PROC] {n} frames... ({fps_now:.1f} fps)", flush=True)
    finally:
        try: dec.stdout.close()
        except: pass
        try: dec.kill()
        except: pass
        try: enc.stdin.close()
        except: pass

    enc_stderr = enc.stderr.read()
    enc.wait()
    if enc.returncode != 0 or write_err:
        msg = enc_stderr.decode()[:500] if enc_stderr else str(write_err)
        raise RuntimeError(f"FFmpeg enc: {msg}")
    dt = time.time() - t_start
    print(f"[PROC] Done: {n} frames in {dt:.1f}s ({n/dt:.1f} fps)", flush=True)
    return final


# ─────────────────────────────────────────────────────────────
# RunPod handler
# ─────────────────────────────────────────────────────────────
def handler(job):
    inp        = job.get('input', {})
    boxes      = inp.get('boxes', [])
    width      = int(inp.get('width', 0))
    height     = int(inp.get('height', 0))
    fps        = float(inp.get('fps', 30.0))
    callback   = inp.get('callback_url', '')
    job_id     = inp.get('job_id', job.get('id', 'unknown'))

    tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    input_path = tmp.name
    tmp.close()

    try:
        if 'video_url' in inp:
            print(f"[DL] {inp['video_url']}", flush=True)
            r = requests.get(inp['video_url'], timeout=300, stream=True)
            r.raise_for_status()
            with open(input_path, 'wb') as f:
                for chunk in r.iter_content(8 * 1024 * 1024):
                    f.write(chunk)
            print(f"[DL] {os.path.getsize(input_path)/1024/1024:.1f} MB", flush=True)
        elif 'video_base64' in inp:
            with open(input_path, 'wb') as f:
                f.write(base64.b64decode(inp['video_base64']))
        else:
            return {'error': 'Niciun video'}

        out = process_video(input_path, boxes, width, height, fps)
        size_mb = os.path.getsize(out) / 1024 / 1024
        print(f"[DONE] {size_mb:.1f} MB → {out}", flush=True)

        if callback:
            print(f"[UPLOAD] POST la {callback}", flush=True)
            with open(out, 'rb') as f:
                resp = requests.post(
                    callback,
                    files={'video': ('result.mp4', f, 'video/mp4')},
                    data={'job_id': job_id},
                    timeout=300,
                )
            if resp.ok:
                print(f"[UPLOAD] OK — {resp.status_code}", flush=True)
                return {'result_uploaded': True, 'job_id': job_id, 'size_mb': round(size_mb, 1)}
            else:
                print(f"[UPLOAD] FAILED {resp.status_code} — fallback base64", flush=True)

        with open(out, 'rb') as f:
            return {'video_base64': base64.b64encode(f.read()).decode()}

    except Exception as e:
        print(f"[ERROR] {e}", flush=True); traceback.print_exc(); return {'error': str(e)}
    finally:
        for p in [input_path, input_path + '_final.mp4']:
            try: os.remove(p)
            except: pass

print("[INIT] Worker pornit.", flush=True)
runpod.serverless.start({'handler': handler})
