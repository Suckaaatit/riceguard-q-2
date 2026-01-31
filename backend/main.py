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
from roboflow import Roboflow
import cloudinary
import cloudinary.uploader
from motor.motor_asyncio import AsyncIOMotorClient

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RiceGuard")

env_path = Path(__file__).parent / ".env"
load_dotenv(env_path, override=True)

# API Keys & Config
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID")
MONGODB_URL = os.getenv("MONGODB_URL")
DB_NAME = os.getenv("DB_NAME", "riceguard_db")
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# Initialize Cloudinary
if CLOUDINARY_CLOUD_NAME:
    try:
        cloudinary.config( 
          cloud_name = CLOUDINARY_CLOUD_NAME, 
          api_key = CLOUDINARY_API_KEY, 
          api_secret = CLOUDINARY_API_SECRET 
        )
    except Exception as e:
        logger.error(f"Cloudinary Init Failed: {e}")

# --- MONGODB CONNECTION CHECK ---
mongo_client = None
db = None
history_collection = None

app = FastAPI(title="RiceGuard API - Ultimate")

@app.on_event("startup")
async def _startup_mongo():
    global mongo_client, db, history_collection

    if not MONGODB_URL:
        logger.warning("⚠️ MONGODB_URL not found in .env - History will NOT be saved.")
        return

    try:
        mongo_client = AsyncIOMotorClient(MONGODB_URL, serverSelectionTimeoutMS=5000)
        await mongo_client.admin.command("ping")
        db = mongo_client[DB_NAME]
        history_collection = db["rice_history"]
        logger.info(f"✅ MongoDB ready (db={DB_NAME}, collection=rice_history)")
    except Exception as e:
        logger.error(f"⚠️ MongoDB Connection Failed ({type(e).__name__}): {e}")
        logger.exception("⚠️ MongoDB Connection Failed")
        mongo_client = None
        db = None
        history_collection = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# AI MODEL LOADER
# -----------------------------------------------------------------------------
_rf_model = None

def get_model():
    global _rf_model
    if _rf_model is None:
        if not ROBOFLOW_API_KEY:
            raise RuntimeError("CRITICAL: ROBOFLOW_API_KEY not found in .env")
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project_id, version_num = ROBOFLOW_MODEL_ID.split("/")
        _rf_model = rf.workspace().project(project_id).version(int(version_num)).model
    return _rf_model

# -----------------------------------------------------------------------------
# DATA SCHEMAS
# -----------------------------------------------------------------------------
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
# INTELLIGENT FILTERS (The Protection Layer)
# -----------------------------------------------------------------------------

def is_digital_screenshot(img) -> bool:
    """Detects digital screenshots (flat colors, UI elements)."""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Histogram Check
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        pixel_count = img.shape[0] * img.shape[1]
        peak_val = np.argmax(hist)
        max_peak = hist[peak_val]
        
        if 10 < peak_val < 245: 
            if max_peak / pixel_count > 0.30: return True
            
        # 2. Smoothness Check
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 3.0: return True 
        
        return False
    except: 
        return False

def contains_qr_code(img) -> bool:
    """Explicitly checks for QR codes to block them."""
    try:
        detector = cv2.QRCodeDetector()
        data, bbox, _ = detector.detectAndDecode(img)
        # If bbox is found, it's a QR code
        if bbox is not None:
            return True
        return False
    except:
        return False

def looks_like_rice(img) -> bool:
    """
    STRICT Organic Check:
    1. Finds contours.
    2. Rejects if too many objects are perfect squares (QR/Pixel art).
    3. Rejects if objects are too random/noisy (Carpet/Shoe).
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        img_area = h * w
        
        # Blur to remove fine texture (carpet fibers)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Adaptive threshold to find blobs
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_grains = 0
        square_shapes = 0
        total_shapes = 0

        for c in cnts:
            area = cv2.contourArea(c)
            # Rice filter: Must be between 0.01% and 1% of image
            if (img_area * 0.0001) < area < (img_area * 0.01):
                total_shapes += 1
                
                # Check Geometry
                x, y, cw, ch = cv2.boundingRect(c)
                aspect_ratio = float(cw) / ch if ch > 0 else 0
                
                # QR Code / Pixel Art Check: Is it a perfect square?
                # Rice is oblong (ratio != 1.0)
                if 0.85 < aspect_ratio < 1.15:
                    square_shapes += 1
                else:
                    valid_grains += 1

        # 1. Not enough grains?
        if valid_grains < 5:
            return False

        # 2. Too many squares? (It's a QR code or Grid)
        if total_shapes > 0 and (square_shapes / total_shapes) > 0.4:
            return False # >40% of objects are squares -> REJECT

        return True
    except:
        return True 

def _to_detections(result_json) -> sv.Detections:
    """Helper to convert Roboflow JSON to Supervision Detections"""
    from_roboflow = getattr(sv.Detections, "from_roboflow", None)
    if callable(from_roboflow):
        return from_roboflow(result_json)
    
    predictions = (result_json or {}).get("predictions", [])
    if not predictions: return sv.Detections.empty()

    xyxy = []
    confidence = []
    class_id = []
    class_names = []
    
    unique_classes = sorted(list(set([p['class'] for p in predictions])))
    name_map = {name: i for i, name in enumerate(unique_classes)}

    for p in predictions:
        x, y, w, h = p['x'], p['y'], p['width'], p['height']
        xyxy.append([x - w/2, y - h/2, x + w/2, y + h/2])
        confidence.append(p['confidence'])
        class_names.append(p['class'])
        class_id.append(name_map[p['class']])

    return sv.Detections(
        xyxy=np.array(xyxy),
        confidence=np.array(confidence),
        class_id=np.array(class_id),
        data={'class_name': np.array(class_names)}
    )

# -----------------------------------------------------------------------------
# CORE LOGIC
# -----------------------------------------------------------------------------
def analyze_logic(image_bytes: bytes) -> AnalysisResult:
    # 1. Decode
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: raise ValueError("Image decoding failed")

    orig_h, orig_w, _ = img.shape
    min_dim = min(orig_w, orig_h)

    # 2. FILTER: Explicit QR Code Check
    if contains_qr_code(img):
        raise HTTPException(status_code=400, detail="QR Code detected. Please upload rice grains.")

    # 3. FILTER: Screenshot Check
    if is_digital_screenshot(img):
        raise HTTPException(status_code=400, detail="Screenshot detected. Please upload a real photo.")

    # 4. FILTER: Strict Shape Analysis (No Squares/Noise)
    if not looks_like_rice(img):
         raise HTTPException(status_code=400, detail="No rice detected. Found mostly noise or synthetic patterns.")

    # 5. Inference (HIGH CONFIDENCE)
    fd, img_path = tempfile.mkstemp(suffix=".jpg")
    try:
        with os.fdopen(fd, 'wb') as tmp: tmp.write(image_bytes)
        
        # 🔥 CONFIDENCE BOOST: 40% (was 10-30%)
        # Only accept if the AI is VERY sure it's rice
        try:
            result = get_model().predict(img_path, confidence=40, overlap=30).json()
        except TypeError:
            result = get_model().predict(img_path, confidence=40).json()
    except Exception as e:
        logger.error(f"Inference Error: {e}")
        raise HTTPException(500, "AI Model Error")
    finally:
        if os.path.exists(img_path): os.remove(img_path)

    detections = _to_detections(result)

    # 6. Extract & Sanity Check Detections
    widths = []
    lengths = []
    counts = {"good": 0, "broken": 0, "chalky": 0, "foreign": 0}
    
    valid_indices = []

    for i in range(len(detections)):
        x1, y1, x2, y2 = detections.xyxy[i]
        w_px = x2 - x1
        h_px = y2 - y1
        
        # GEOMETRIC SANITY CHECKS:
        
        # A. Too Big? (Rice isn't half the image size)
        if w_px > (orig_w * 0.2) or h_px > (orig_h * 0.2):
            continue 
            
        # B. Too Small? (Noise)
        if w_px < 10 or h_px < 10:
            continue

        # C. Aspect Ratio Check (Rice is oblong)
        # Rejects squares (0.9 to 1.1 ratio) which usually indicate pixels/QR blocks
        ratio = w_px / h_px if h_px > 0 else 0
        if 0.85 < ratio < 1.15:
            continue # Skip perfectly square detections

        valid_indices.append(i)
        
        cls = str(detections.data['class_name'][i]).lower()
        if "broken" in cls: counts["broken"] += 1
        elif "chalky" in cls: counts["chalky"] += 1
        elif "foreign" in cls: counts["foreign"] += 1
        else: counts["good"] += 1 

        widths.append(min(w_px, h_px))
        lengths.append(max(w_px, h_px))
        
    if len(valid_indices) < len(detections):
        try:
            detections = detections[np.array(valid_indices)]
        except:
             pass # Handle empty array edge case

    # 7. Minimum Grain Count Check
    # Need at least 8 confirmed non-square grains to pass
    if len(valid_indices) < 8:
        _, buf = cv2.imencode(".jpg", img)
        return AnalysisResult(
            total_grains=0, whole_grains=0, broken_grains=0, 
            chalky_grains=0, foreign_matter=0, 
            avg_width_mm=0.0, avg_length_mm=0.0,
            visualization=base64.b64encode(buf).decode(),
            timestamp=datetime.now().strftime("%I:%M %p"),
            warnings=["No rice grains found (objects were noise/squares)."]
        )

    # 8. Distance & Calibration Logic
    avg_w = 0.0
    avg_l = 0.0
    
    if len(widths) > 0:
        median_px_width = float(np.median(widths))
        width_ratio = median_px_width / float(min_dim)

        if width_ratio < 0.005:
            raise HTTPException(400, "Camera too far! Grains too small. Move to 10-12cm.")
        if width_ratio > 0.035:
            raise HTTPException(400, "Camera too close! Move back.")

        estimated_px_to_mm = 2.2 / median_px_width
        avg_w = round(np.median(widths) * estimated_px_to_mm, 2)
        avg_l = round(np.median(lengths) * estimated_px_to_mm, 2)

    # 9. Visualization
    vis_b64 = ""
    try:
        box_annotator = sv.BoxAnnotator(thickness=2)
        annotated = img.copy()
        annotated = box_annotator.annotate(scene=annotated, detections=detections)
        _, buf = cv2.imencode(".jpg", annotated)
        vis_b64 = base64.b64encode(buf).decode()
    except:
        _, buf = cv2.imencode(".jpg", img)
        vis_b64 = base64.b64encode(buf).decode()

    # 10. Cloudinary Upload
    upload_url = ""
    if CLOUDINARY_CLOUD_NAME:
        try:
            resp = cloudinary.uploader.upload(
                f"data:image/jpg;base64,{vis_b64}",
                folder="rice_guard_history", 
                tags=["rice_history"]
            )
            upload_url = resp.get("secure_url")
        except Exception as e:
            logger.error(f"Cloudinary upload failed: {e}")

    return AnalysisResult(
        total_grains=len(valid_indices),
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
# ENDPOINTS
# -----------------------------------------------------------------------------

@app.post("/analyze", response_model=AnalysisResult)
async def analyze(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    
    try:
        contents = await file.read()
        result = await run_in_threadpool(analyze_logic, contents)
    except HTTPException as he:
        raise he
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Analysis failed: {str(e)}")

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
        except Exception:
            logger.exception("DB Insert failed")

    return result

@app.get("/history")
async def get_history():
    if history_collection is None: return []
    try:
        cursor = history_collection.find().sort("created_at", -1).limit(20)
        items = []
        async for doc in cursor:
            items.append({
                "id": str(doc.get("_id")),
                "url": doc.get("image_url", ""),
                "timestamp": doc.get("display_time", ""),
                "stats": {
                    "total": doc.get("total_grains", 0),
                    "whole": doc.get("whole_grains", 0),
                    "broken": doc.get("broken_grains", 0)
                }
            })
        return items
    except Exception:
        return []

@app.get("/health")
def health(): return {"status": "ok"}