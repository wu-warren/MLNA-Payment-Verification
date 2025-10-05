import os, io, re
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np

# ---- EasyOCR (text recognizer) ----
import easyocr
_EASY = easyocr.Reader(['en'], gpu=False)

def ocr_text(pil_image: Image.Image) -> str:
    arr = np.array(pil_image.convert("RGB"))
    lines = _EASY.readtext(arr, detail=0, paragraph=True)
    return "\n".join(lines)

def parse_amount(text: str):
    m = re.search(r'\$\s*([0-9]+(?:\.[0-9]{2})?)', text)
    return float(m.group(1)) if m else None

def parse_handle(text: str):
    m = re.search(r'@([A-Za-z0-9_.]+)', text)
    return m.group(1) if m else None

def parse_date(text: str):
    m = re.search(r'(\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})', text)
    return m.group(1) if m else None

# ---- YOLO (field detector) ----
from ultralytics import YOLO
_DET = None
CLS = {0: "amount", 1: "date", 2: "handle"}

def load_detector():
    """Lazy-load YOLO weights from env var DETECTOR_WEIGHTS (e.g., models/detector_v1/best.pt)."""
    global _DET
    if _DET is None:
        weights = os.getenv("DETECTOR_WEIGHTS", "models/detector_v1/best.pt")
        if not os.path.exists(weights):
            raise RuntimeError(f"Detector weights not found at {weights}. Set DETECTOR_WEIGHTS env var.")
        _DET = YOLO(weights)
    return _DET

def detect_and_read(image: Image.Image, conf: float = 0.35):
    model = load_detector()
    H, W = image.height, image.width
    results = model.predict(source=image, conf=conf, verbose=False)[0]

    fields = {"amount": None, "date": None, "handle": None}
    for box, cls in zip(results.boxes.xywh, results.boxes.cls):
        x, y, w, h = box.tolist()
        name = CLS.get(int(cls.item()), "other")
        # add a small margin around the box to help OCR
        margin = 0.12
        x0 = max(0, int((x - w/2) - w*margin))
        y0 = max(0, int((y - h/2) - h*margin))
        x1 = min(W, int((x + w/2) + w*margin))
        y1 = min(H, int((y + h/2) + h*margin))
        crop = image.crop((x0, y0, x1, y1))
        text = ocr_text(crop)

        if name == "amount" and fields["amount"] is None:
            fields["amount"] = parse_amount(text)
        elif name == "handle" and fields["handle"] is None:
            fields["handle"] = parse_handle(text)
        elif name == "date" and fields["date"] is None:
            fields["date"] = parse_date(text)
    return fields

# ---- FastAPI app ----
app = FastAPI(title="Venmo Verify (YOLO + EasyOCR)")

@app.get("/healthz")
def health():
    return {"ok": True}

@app.post("/verify-local")
def verify_local(image_path: str, expected_amount: float | None = None, expected_handle: str | None = None):
    """
    Step-1 test endpoint: pass a LOCAL image path to try detection + OCR.
    Example:
      curl -X POST "http://127.0.0.1:8000/verify-local?image_path=./sample.jpg&expected_amount=25.00&expected_handle=mygroup"
    """
    if not os.path.exists(image_path):
        raise HTTPException(status_code=400, detail=f"image not found: {image_path}")
    try:
        im = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"failed to open image: {e}")

    try:
        conf = float(os.getenv("DETECTOR_CONF", "0.35"))
        fields = detect_and_read(im, conf=conf)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"detector error: {e}")

    # simple decision logic (tune later)
    decision, reason = "manual_review", "incomplete_fields"
    if fields["amount"] is not None:
        ok_amt = (expected_amount is None) or (abs(fields["amount"] - expected_amount) <= 0.01)
        ok_handle = (expected_handle is None) or (fields["handle"] == expected_handle)
        if ok_amt and ok_handle:
            decision, reason = "approved", "matched_fields"
        else:
            decision, reason = "manual_review", "mismatch"
    if all(v is None for v in fields.values()):
        decision, reason = "rejected", "no_fields_found"

    return JSONResponse({"decision": decision, "reason": reason, "fields": fields})

# Placeholder for your future Google Drive flow:
@app.post("/verify-payment")
def verify_payment(fileId: str):
    raise HTTPException(status_code=501, detail="TODO: implement Google Drive download by fileId")
