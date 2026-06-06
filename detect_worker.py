#!/usr/bin/env python3
"""
detect_worker.py — auto-detectare text/captioane dintr-un frame video
Utilizare: python detect_worker.py <cale_imagine>
Output stdout: JSON {"width": W, "height": H, "boxes": [{x,y,w,h}, ...]}
"""
import sys
import json
import cv2
import numpy as np

PAD = 8


def merge_nearby(boxes, gap=25):
    if not boxes:
        return []
    changed = True
    while changed:
        changed, merged, used = False, [], [False] * len(boxes)
        for i, a in enumerate(boxes):
            if used[i]:
                continue
            g = [a]
            for j, b in enumerate(boxes):
                if i == j or used[j]:
                    continue
                v_ov = min(a['y'] + a['h'], b['y'] + b['h']) - max(a['y'], b['y'])
                if v_ov > 0:
                    h_dist = max(0, max(a['x'], b['x']) - min(a['x'] + a['w'], b['x'] + b['w']))
                    if h_dist <= gap:
                        g.append(b)
                        used[j] = True
                        changed = True
            x  = min(p['x'] for p in g)
            y  = min(p['y'] for p in g)
            x2 = max(p['x'] + p['w'] for p in g)
            y2 = max(p['y'] + p['h'] for p in g)
            merged.append({'x': x, 'y': y, 'w': x2 - x, 'h': y2 - y})
            used[i] = True
        boxes = merged
    return boxes


def detect_easyocr(img_bgr, h, w):
    import easyocr
    reader = easyocr.Reader(['en', 'ro'], gpu=False, verbose=False)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = reader.readtext(img_rgb, detail=1)
    boxes = []
    for (bbox, text, conf) in results:
        if conf < 0.2 or not str(text).strip():
            continue
        pts = np.array(bbox, dtype=np.int32)
        x, y, bw, bh = cv2.boundingRect(pts)
        boxes.append({
            'x': int(max(0, x - PAD)),
            'y': int(max(0, y - PAD)),
            'w': int(min(w - max(0, x - PAD), bw + 2 * PAD)),
            'h': int(min(h - max(0, y - PAD), bh + 2 * PAD)),
        })
    sys.stderr.write(f'[DETECT] easyocr: {len(boxes)} zone\n')
    return boxes


def detect_pytesseract(img_bgr, h, w):
    import pytesseract
    data = pytesseract.image_to_data(img_bgr, output_type=pytesseract.Output.DICT)
    boxes = []
    for i in range(len(data['text'])):
        t = str(data['text'][i]).strip()
        c = int(data['conf'][i])
        if not t or c < 30:
            continue
        x  = max(0, data['left'][i] - PAD)
        y  = max(0, data['top'][i] - PAD)
        bw = min(w - x, data['width'][i] + 2 * PAD)
        bh = min(h - y, data['height'][i] + 2 * PAD)
        if bw > 0 and bh > 0:
            boxes.append({'x': x, 'y': y, 'w': bw, 'h': bh})
    sys.stderr.write(f'[DETECT] pytesseract: {len(boxes)} zone\n')
    return boxes


def main():
    if len(sys.argv) < 2:
        print(json.dumps({'width': 0, 'height': 0, 'boxes': []}))
        return

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(json.dumps({'width': 0, 'height': 0, 'boxes': []}))
        return

    h, w = img.shape[:2]
    boxes = []

    try:
        boxes = detect_easyocr(img, h, w)
    except Exception as e:
        sys.stderr.write(f'[DETECT] easyocr err: {e}\n')
        try:
            boxes = detect_pytesseract(img, h, w)
        except Exception as e2:
            sys.stderr.write(f'[DETECT] pytesseract err: {e2}\n')

    boxes = merge_nearby(boxes)
    print(json.dumps({'width': w, 'height': h, 'boxes': boxes}))


if __name__ == '__main__':
    main()
