from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
import numpy as np
import os
import base64
import cv2
import traceback
import tempfile
import supervision as sv
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv
from pathlib import Path
import cloudinary
import cloudinary.uploader
import cloudinary.api
from datetime import datetime

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path, override=True)

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID")

# --- CALIBRATION FACTORS ---
PX_TO_MM = float(os.getenv("PX_TO_MM", "0.19")) 
WIDTH_CORRECTION_FACTOR = float(os.getenv("WIDTH_CORRECTION_FACTOR", "0.30")) 
LENGTH_CORRECTION_FACTOR = float(os.getenv("LENGTH_CORRECTION_FACTOR", "1.0"))

# --- CLOUDINARY CONFIG ---
cloudinary.config( 
  cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME", "dbqswhaen"), 
  api_key = os.getenv("CLOUDINARY_API_KEY", "784198114597741"), 
  api_secret = os.getenv("CLOUDINARY_API_SECRET", "kbg3ob9cacNjDUjfCai0Dp3h_eo") 
)

app = FastAPI(title="RiceGuard API - Lenient Mode")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisResult(BaseModel):
    total_grains: int
    whole_grains: int       
    broken_grains: int
    chalky_grains: int
    foreign_matter: int
    avg_width_mm: float
    avg_length_mm: float
    visualization: str
    image_url: str = None
    timestamp: str = None

# --- CLIENT INITIALIZATION ---
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

CONFIDENCE_THRESHOLD = 0.10
IOU_THRESHOLD = 0.3
try:
    object.__setattr__(CLIENT.inference_configuration, "confidence_threshold", float(CONFIDENCE_THRESHOLD))
    object.__setattr__(CLIENT.inference_configuration, "iou_threshold", float(IOU_THRESHOLD))
except Exception:
    pass

def parse_roboflow_manually(json_result):
    preds = (json_result or {}).get("predictions", [])
    if not preds: 
        return sv.Detections.empty()

    xyxy = []
    class_id = []
    confidence = []
    class_names = []
    
    raw_names = [str(p.get("class", "")) for p in preds if p.get("class")]
    unique_classes = sorted(list(set(raw_names)))
    class_map = {name: i for i, name in enumerate(unique_classes)}

    for p in preds:
        if not isinstance(p, dict): continue
        x, y, w, h = p.get("x"), p.get("y"), p.get("width"), p.get("height")
        if any(v is None for v in [x, y, w, h]): continue
        
        # Only filter extremely tiny noise
        if w < 5 or h < 5: continue 

        x1, y1, x2, y2 = x - w/2, y - h/2, x + w/2, y + h/2
        xyxy.append([x1, y1, x2, y2])
        
        cls_name = str(p.get("class", "") or "")
        class_id.append(int(class_map.get(cls_name, 0)))
        confidence.append(float(p.get("confidence", 0.0)))
        class_names.append(cls_name)

    if not xyxy: 
        return sv.Detections.empty()

    return sv.Detections(
        xyxy=np.array(xyxy, dtype=float),
        class_id=np.array(class_id, dtype=int),
        confidence=np.array(confidence, dtype=float),
        data={"class_name": np.array(class_names, dtype=object)}
    )

def analyze_logic(image_bytes: bytes) -> AnalysisResult:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: raise ValueError("Image decoding failed")

    # 1. SQUARE CROP
    h, w, _ = img.shape
    if w != h:
        min_dim = min(w, h)
        start_x = (w - min_dim) // 2
        start_y = (h - min_dim) // 2
        img = img[start_y:start_y+min_dim, start_x:start_x+min_dim]

    # 2. INFERENCE
    fd, img_path = tempfile.mkstemp(suffix=".jpg")
    try:
        with os.fdopen(fd, 'wb') as tmp: 
            _, encoded_img = cv2.imencode(".jpg", img)
            tmp.write(encoded_img.tobytes())
        
        result = CLIENT.infer(
            img_path,
            model_id=ROBOFLOW_MODEL_ID,
        )
    finally:
        if os.path.exists(img_path): os.remove(img_path)

    detections = parse_roboflow_manually(result)
    
    # --- SUPER LENIENT FILTERING ---
    # We basically accept almost everything now to stop the "0 detections" error
    valid_indices = []
    final_widths = []
    final_lengths = []
    
    # RELAXED LIMITS
    rice_max_width = 6.0   # Was 3.2 -> Increased to avoid deleting big rice
    rice_min_length = 1.0  # Was 2.5 -> Decreased to keep broken tips

    for i in range(len(detections)):
        x1, y1, x2, y2 = detections.xyxy[i]
        cls_name = str(detections.data['class_name'][i]).lower()
        
        w_px = min(x2 - x1, y2 - y1)
        h_px = max(x2 - x1, y2 - y1)
        
        w_mm = w_px * PX_TO_MM * WIDTH_CORRECTION_FACTOR
        h_mm = h_px * PX_TO_MM * LENGTH_CORRECTION_FACTOR
        
        # LOGIC:
        # 1. If it's foreign/stone -> KEEP IT (unless < 0.5mm noise)
        if "foreign" in cls_name or "stone" in cls_name or "shell" in cls_name:
            if w_mm > 0.5: valid_indices.append(i)
            continue 

        # 2. If it's Rice -> CHECK SIZE (But gently)
        if w_mm > rice_max_width: continue  # Only delete if HUGE (>6mm width)
        if h_mm < rice_min_length: continue # Only delete if TINY (<1mm)

        # 3. Ratio Check -> Only for clearly whole rice
        ratio = h_mm / w_mm if w_mm > 0 else 0
        if "broken" not in cls_name and ratio < 1.1: 
            # It's a perfect square/circle. Likely a pebble, but let's be safe.
            # If the model says "Rice", we trust the model more than the math now.
            pass 

        valid_indices.append(i)
        final_widths.append(w_mm)
        final_lengths.append(h_mm)

    # --- HANDLE EMPTY RESULT ---
    if not valid_indices:
        _, buf = cv2.imencode(".jpg", img)
        vis_b64 = base64.b64encode(buf).decode()
        # Returns 0 count, but NO ERROR
        return AnalysisResult(
            total_grains=0, whole_grains=0, broken_grains=0, chalky_grains=0, foreign_matter=0, 
            avg_width_mm=0.0, avg_length_mm=0.0, visualization=vis_b64, image_url=None, timestamp=datetime.now().strftime("%Y-%m-%d %H:%M")
        )
    
    detections = detections[np.array(valid_indices)]

    # --- STATS ---
    avg_w = round(np.median(final_widths), 2) if final_widths else 0.0
    avg_l = round(np.median(final_lengths), 2) if final_lengths else 0.0
    
    # ⚠️ REMOVED ALL WARNING PRINTS/LOGIC HERE
    
    counts = {"good": 0, "broken": 0, "chalky": 0, "foreign": 0}
    
    for i in range(len(detections)):
        cls = str(detections.data['class_name'][i]).lower()
        if "foreign" in cls or "stone" in cls: counts["foreign"] += 1
        elif "broken" in cls: counts["broken"] += 1
        elif "chalky" in cls: counts["chalky"] += 1
        elif "whole" in cls or "good" in cls: counts["good"] += 1
        else: counts["foreign"] += 1 

    # --- VISUALIZATION ---
    try:
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=5)
        
        labels = []
        for k in range(len(detections)):
             # Show class + width to help debug calibration
             w_disp = (min(detections.xyxy[k][2]-detections.xyxy[k][0], detections.xyxy[k][3]-detections.xyxy[k][1]) * PX_TO_MM * WIDTH_CORRECTION_FACTOR)
             labels.append(f"{detections.data['class_name'][k]} {w_disp:.1f}mm")

        annotated = img.copy()
        annotated = box_annotator.annotate(scene=annotated, detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
        
        _, buf = cv2.imencode(".jpg", annotated)
        vis_b64 = base64.b64encode(buf).decode()
    except Exception as e:
        print(f"Vis Error: {e}")
        _, buf = cv2.imencode(".jpg", img)
        vis_b64 = base64.b64encode(buf).decode()

    # --- CLOUDINARY UPLOAD ---
    upload_url = ""
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    try:
        meta_context = f"total={len(detections)}|whole={counts['good']}|broken={counts['broken']}|chalky={counts['chalky']}"
        response = cloudinary.uploader.upload(
            f"data:image/jpg;base64,{vis_b64}",
            folder="rice_guard_history",
            context=meta_context,
            tags=["rice_history"]
        )
        upload_url = response.get("secure_url")
    except Exception as e:
        print(f"Cloudinary Upload Failed: {e}")

    return AnalysisResult(
        total_grains=len(detections),
        whole_grains=counts["good"],
        broken_grains=counts["broken"],
        chalky_grains=counts["chalky"],
        foreign_matter=counts["foreign"],
        avg_width_mm=avg_w,
        avg_length_mm=avg_l,
        visualization=vis_b64,
        image_url=upload_url,
        timestamp=timestamp_str
    )

@app.post("/analyze", response_model=AnalysisResult)
async def analyze(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    try:
        contents = await file.read()
        return await run_in_threadpool(analyze_logic, contents)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/history")
def get_history():
    try:
        resources = cloudinary.api.resources(
            type="upload", 
            prefix="rice_guard_history", 
            max_results=10, 
            context=True,
            direction="desc"
        )
        history_items = []
        for res in resources.get('resources', []):
            ctx = res.get('context', {}).get('custom', {})
            history_items.append({
                "url": res.get("secure_url"),
                "created_at": res.get("created_at"),
                "context": ctx
            })
        return history_items
    except Exception:
        return []

@app.get("/health")
def health():
    return {"status": "ok"}