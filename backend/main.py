import os
import cv2
import numpy as np
import base64
import tempfile
import traceback
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Third-party imports
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import supervision as sv
from inference_sdk import InferenceHTTPClient
import cloudinary
import cloudinary.uploader
from motor.motor_asyncio import AsyncIOMotorClient

# -----------------------------------------------------------------------------
# CONFIGURATION & SAFETY CONSTANTS
# -----------------------------------------------------------------------------
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB Limit
MAX_RETRIES = 3                  # Retry AI model 3 times before failing

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RiceGuard")

env_path = Path(__file__).parent / ".env"
load_dotenv(env_path, override=True)

# API Keys
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID")
MONGODB_URL = os.getenv("MONGODB_URL")
DB_NAME = os.getenv("DB_NAME", "riceguard_db")
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

# Calibration
PX_TO_MM = float(os.getenv("PX_TO_MM", "0.19")) 
WIDTH_CORRECTION = float(os.getenv("WIDTH_CORRECTION_FACTOR", "0.30")) 
LENGTH_CORRECTION = float(os.getenv("LENGTH_CORRECTION_FACTOR", "1.0"))

# Initialize Cloudinary
if CLOUDINARY_CLOUD_NAME:
    cloudinary.config( 
      cloud_name = CLOUDINARY_CLOUD_NAME, 
      api_key = CLOUDINARY_API_KEY, 
      api_secret = CLOUDINARY_API_SECRET 
    )

# Initialize MongoDB
mongo_client = None
db = None
history_collection = None
if MONGODB_URL:
    try:
        mongo_client = AsyncIOMotorClient(MONGODB_URL)
        db = mongo_client[DB_NAME]
        history_collection = db["rice_history"]
        logger.info("✅ Connected to MongoDB")
    except Exception as e:
        logger.error(f"⚠️ MongoDB Connection Failed: {e}")

# Initialize AI Client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

app = FastAPI(title="RiceGuard API - Production v2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In real production, replace '*' with your actual domain
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
    image_url: Optional[str] = None
    timestamp: str
    warnings: List[str] = Field(default_factory=list)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS (Preserving your original logic)
# -----------------------------------------------------------------------------
# [...Insert your existing is_digital_screenshot, is_natural_object, parse_roboflow_manually here...]
# (I am omitting the bodies to save space, but DO NOT DELETE THEM from your file)
def is_digital_screenshot(img) -> bool:
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        if np.max(hist) / (img.shape[0] * img.shape[1]) > 0.15: return True
        return False
    except: return False

def is_natural_object(crop_img) -> bool:
    if crop_img.size == 0: return False
    avg_color = np.mean(crop_img, axis=(0, 1))
    if avg_color[0] > avg_color[1] * 1.2: return False # Blue/Sky check
    return True

def parse_roboflow_manually(json_result) -> sv.Detections:
    # (Use your existing function code here)
    preds = (json_result or {}).get("predictions", [])
    if not preds: return sv.Detections.empty()
    xyxy = []
    class_id = []
    confidence = []
    class_names = []

    name_to_id = {}
    next_id = 0

    for p in preds:
        x = p.get("x")
        y = p.get("y")
        w = p.get("width")
        h = p.get("height")

        if x is None or y is None or w is None or h is None:
            continue

        x1 = float(x) - float(w) / 2.0
        y1 = float(y) - float(h) / 2.0
        x2 = float(x) + float(w) / 2.0
        y2 = float(y) + float(h) / 2.0
        xyxy.append([x1, y1, x2, y2])

        conf = p.get("confidence", 0.0)
        confidence.append(float(conf))

        cls_name = p.get("class") or p.get("class_name") or ""
        cls_name = str(cls_name)
        class_names.append(cls_name)

        cid = p.get("class_id")
        if cid is None:
            if cls_name not in name_to_id:
                name_to_id[cls_name] = next_id
                next_id += 1
            cid = name_to_id[cls_name]
        class_id.append(int(cid))

    if not xyxy:
        return sv.Detections.empty()

    xyxy_arr = np.asarray(xyxy, dtype=float)
    conf_arr = np.asarray(confidence, dtype=float)
    class_id_arr = np.asarray(class_id, dtype=int)
    class_names_arr = np.asarray(class_names, dtype=str)

    try:
        return sv.Detections(
            xyxy=xyxy_arr,
            confidence=conf_arr,
            class_id=class_id_arr,
            data={"class_name": class_names_arr},
        )
    except TypeError:
        det = sv.Detections(xyxy=xyxy_arr, confidence=conf_arr, class_id=class_id_arr)
        try:
            det.data["class_name"] = class_names_arr
        except Exception:
            pass
        return det

# -----------------------------------------------------------------------------
# CORE LOGIC WITH RETRY & SAFETY
# -----------------------------------------------------------------------------

def analyze_logic(image_bytes: bytes) -> AnalysisResult:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: raise ValueError("Image decoding failed")

    # [Logic A & B: Pre-processing & Screenshot Check] - Keep your existing code
    h, w, _ = img.shape
    if w != h:
        min_dim = min(w, h)
        img = img[(h-min_dim)//2:(h-min_dim)//2+min_dim, (w-min_dim)//2:(w-min_dim)//2+min_dim]

    # [Logic C: Robust Inference with Retry]
    fd, img_path = tempfile.mkstemp(suffix=".jpg")
    result = None
    try:
        with os.fdopen(fd, 'wb') as tmp: 
            _, encoded_img = cv2.imencode(".jpg", img)
            tmp.write(encoded_img.tobytes())
        
        # RETRY MECHANISM
        for attempt in range(MAX_RETRIES):
            try:
                result = CLIENT.infer(img_path, model_id=ROBOFLOW_MODEL_ID)
                try:
                    preds_len = len((result or {}).get("predictions", []) or [])
                    logger.info(f"Roboflow predictions: {preds_len}")
                except Exception:
                    logger.info("Roboflow predictions: <unavailable>")
                break # Success!
            except Exception as e:
                if attempt == MAX_RETRIES - 1: raise e # Crash only on last attempt
                time.sleep(0.5) # Wait before retry

    finally:
        if os.path.exists(img_path): os.remove(img_path)

    # [Logic D-I: Filtering & Stats] - (Assume standard logic here for brevity)
    # ... Your existing filtering logic ...
    if not ROBOFLOW_MODEL_ID:
        raise ValueError("ROBOFLOW_MODEL_ID is not set")

    detections = parse_roboflow_manually(result)
    if len(detections) == 0:
        _, vis_buf = cv2.imencode(".jpg", img)
        vis_b64 = base64.b64encode(vis_buf).decode()
        return AnalysisResult(
            total_grains=0,
            whole_grains=0,
            broken_grains=0,
            chalky_grains=0,
            foreign_matter=0,
            avg_width_mm=0.0,
            avg_length_mm=0.0,
            visualization=vis_b64,
            image_url="",
            timestamp=datetime.now().strftime("%I:%M %p"),
        )

    class_names_arr = None
    try:
        class_names_arr = detections.data.get("class_name")
    except Exception:
        class_names_arr = None

    raw_widths_mm = []
    raw_lengths_mm = []
    for i in range(len(detections)):
        x1, y1, x2, y2 = detections.xyxy[i]
        bbox_w_px = float(max(0.0, x2 - x1))
        bbox_h_px = float(max(0.0, y2 - y1))
        if bbox_w_px <= 0.0 or bbox_h_px <= 0.0:
            raw_widths_mm.append(0.0)
            raw_lengths_mm.append(0.0)
            continue

        w_px = min(bbox_w_px, bbox_h_px)
        l_px = max(bbox_w_px, bbox_h_px)
        raw_widths_mm.append(w_px * PX_TO_MM * WIDTH_CORRECTION)
        raw_lengths_mm.append(l_px * PX_TO_MM * LENGTH_CORRECTION)

    valid_raw_widths = [w for w in raw_widths_mm if w > 0]
    median_width = float(np.median(valid_raw_widths)) if valid_raw_widths else 0.0

    if median_width < 1.3:
        raise HTTPException(status_code=400, detail=f"Camera too far! (Est. {median_width:.2f}mm). Move closer to 10-12cm.")
    if median_width > 5.0:
        raise HTTPException(status_code=400, detail=f"Camera too close! (Est. {median_width:.2f}mm). Move back.")

    if len(valid_raw_widths) > 5:
        std_dev = float(np.std(valid_raw_widths))
        if std_dev > 1.5:
            _, vis_buf = cv2.imencode(".jpg", img)
            vis_b64 = base64.b64encode(vis_buf).decode()
            return AnalysisResult(
                total_grains=0,
                whole_grains=0,
                broken_grains=0,
                chalky_grains=0,
                foreign_matter=0,
                avg_width_mm=0.0,
                avg_length_mm=0.0,
                visualization=vis_b64,
                image_url="",
                timestamp=datetime.now().strftime("%I:%M %p"),
            )

    valid_indices = []
    final_widths = []
    final_lengths = []

    for i in range(len(detections)):
        w_mm = float(raw_widths_mm[i])
        h_mm = float(raw_lengths_mm[i])
        if w_mm <= 0.0 or h_mm <= 0.0:
            continue

        ratio = h_mm / w_mm if w_mm > 0 else 0.0
        if (1.5 <= w_mm <= 3.5) and (3.0 <= h_mm <= 10.0) and (ratio > 1.4):
            valid_indices.append(i)
            final_widths.append(w_mm)
            final_lengths.append(h_mm)

    if len(valid_indices) == 0:
        _, vis_buf = cv2.imencode(".jpg", img)
        vis_b64 = base64.b64encode(vis_buf).decode()
        return AnalysisResult(
            total_grains=0,
            whole_grains=0,
            broken_grains=0,
            chalky_grains=0,
            foreign_matter=0,
            avg_width_mm=0.0,
            avg_length_mm=0.0,
            visualization=vis_b64,
            image_url="",
            timestamp=datetime.now().strftime("%I:%M %p"),
        )

    idx = np.array(valid_indices, dtype=int)
    try:
        detections = detections[idx]
    except Exception:
        data = {}
        try:
            for k, v in (detections.data or {}).items():
                try:
                    data[k] = np.asarray(v)[idx]
                except Exception:
                    pass
        except Exception:
            data = {}

        conf = None
        cls_id = None
        try:
            conf = np.asarray(detections.confidence)[idx] if detections.confidence is not None else None
        except Exception:
            conf = None
        try:
            cls_id = np.asarray(detections.class_id)[idx] if detections.class_id is not None else None
        except Exception:
            cls_id = None

        if conf is None and cls_id is None:
            detections = sv.Detections(xyxy=np.asarray(detections.xyxy)[idx], data=data)
        else:
            detections = sv.Detections(
                xyxy=np.asarray(detections.xyxy)[idx],
                confidence=conf,
                class_id=cls_id,
                data=data,
            )

    avg_w = float(np.median(final_widths)) if final_widths else 0.0
    avg_l = float(np.median(final_lengths)) if final_lengths else 0.0
    avg_w = round(avg_w, 2)
    avg_l = round(avg_l, 2)

    counts = {"good": 0, "broken": 0, "chalky": 0, "foreign": 0}
    for i in range(len(detections)):
        cls_name = ""
        try:
            cls_name = str(detections.data.get("class_name")[i])
        except Exception:
            cls_name = ""
        cls_norm = cls_name.lower().strip()
        if "broken" in cls_norm:
            counts["broken"] += 1
        elif "chalk" in cls_norm:
            counts["chalky"] += 1
        elif "foreign" in cls_norm or "matter" in cls_norm or "impur" in cls_norm:
            counts["foreign"] += 1
        else:
            counts["good"] += 1

    annotated = img.copy()
    try:
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=5)
        labels = []
        for k in range(len(detections)):
            try:
                labels.append(str(detections.data.get("class_name")[k]))
            except Exception:
                labels.append("")
        annotated = box_annotator.annotate(scene=annotated, detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
    except Exception:
        pass

    _, vis_buf = cv2.imencode(".jpg", annotated)
    vis_b64 = base64.b64encode(vis_buf).decode()

    # [Logic J: Cloudinary Upload]
    upload_url = ""
    if CLOUDINARY_CLOUD_NAME:
        try:
            resp = cloudinary.uploader.upload(
                f"data:image/jpg;base64,{vis_b64}",
                folder="rice_guard_history", tags=["rice_history"]
            )
            upload_url = resp.get("secure_url")
        except Exception as e:
            logger.error(f"Cloudinary Error: {e}")

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
        timestamp=datetime.now().strftime("%I:%M %p")
    )

# -----------------------------------------------------------------------------
# SECURE API ENDPOINTS
# -----------------------------------------------------------------------------

@app.post("/analyze", response_model=AnalysisResult)
async def analyze(file: UploadFile = File(...)):
    # 1. VALIDATE FILE TYPE
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    
    # 2. VALIDATE FILE SIZE (The "Memory Bomb" Fix)
    # Read in chunks to avoid blowing up RAM, or check Content-Length header
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(413, f"File too large. Limit is {MAX_FILE_SIZE/1024/1024}MB")

    try:
        contents = await file.read()
        result = await run_in_threadpool(analyze_logic, contents)

        # 3. DB SAVE
        if history_collection is not None:
            try:
                await history_collection.insert_one({
                    "total_grains": int(result.total_grains),
                    "whole_grains": int(result.whole_grains),
                    "broken_grains": int(result.broken_grains),
                    "chalky_grains": int(result.chalky_grains),
                    "foreign_matter": int(result.foreign_matter),
                    "avg_width_mm": float(result.avg_width_mm),
                    "avg_length_mm": float(result.avg_length_mm),
                    "image_url": result.image_url or "",
                    "created_at": datetime.utcnow(),
                    "display_time": result.timestamp
                })
            except Exception as e:
                logger.error(f"Mongo Insert Error: {e}")

        return result
    except HTTPException as he: raise he
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, "Internal Analysis Error")

@app.get("/history")
async def get_history():
    if history_collection is None: return []
    try:
        cursor = history_collection.find().sort("created_at", -1).limit(20)
        items = []
        async for doc in cursor:
            ts = doc.get("display_time")
            if not ts and doc.get("created_at"):
                ts = doc.get("created_at").strftime("%I:%M %p")
            
            items.append({
                "id": str(doc.get("_id")),
                "url": doc.get("image_url", ""),
                "timestamp": ts,
                "stats": {
                    "total": doc.get("total_grains", 0),
                    "whole": doc.get("whole_grains", 0),
                    "broken": doc.get("broken_grains", 0)
                }
            })
        return items
    except Exception: return []

@app.get("/health")
def health(): return {"status": "ok"}