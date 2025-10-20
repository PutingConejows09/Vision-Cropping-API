import os, base64, tempfile, traceback
import numpy as np
import cv2
from flask import Flask, request, jsonify
from PIL import Image
import io

app = Flask(__name__)

# Optional: simple API key protection
API_KEY = os.getenv("API_KEY")

def require_api_key():
    if not API_KEY:
        return True
    return request.headers.get("X-API-KEY") == API_KEY


def clamp(v, lo, hi):
    """Keep value v within [lo, hi]."""
    return max(lo, min(hi, v))


# OpenCV auto-detection - NO Vision API, NO COST! 💰
def crop_receipt_auto(image_input, padding=10, is_base64=False):
    """
    Auto-detects receipt boundaries using OpenCV image processing.
    NO Vision API calls = NO EXTRA COST!
    
    Args:
        image_input: file path or base64 string
        padding: extra pixels around detected area
        is_base64: True if image_input is base64 string
    
    Returns:
        base64 encoded cropped JPEG or None
    """
    try:
        # Load image
        if is_base64:
            img_data = base64.b64decode(image_input)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(image_input)
        
        if img is None:
            print("❌ Failed to load image")
            return None
        
        original_h, original_w = img.shape[:2]
        print(f"📐 Original image size: {original_w}x{original_h}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges to connect nearby contours
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("⚠️ No contours found, returning original image")
            # Return original if no contours found
            _, buffer = cv2.imencode('.jpg', img)
            b64 = base64.b64encode(buffer).decode('utf-8')
            return b64
        
        # Find the largest contour (likely the receipt)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        print(f"📦 Detected receipt area: x={x}, y={y}, w={w}, h={h}")
        
        # Apply padding and clamp to image boundaries
        x = clamp(x - padding, 0, original_w)
        y = clamp(y - padding, 0, original_h)
        x2 = clamp(x + w + 2*padding, 0, original_w)
        y2 = clamp(y + h + 2*padding, 0, original_h)
        
        # Crop the image
        cropped = img[y:y2, x:x2]
        
        print(f"✂️ Cropped size: {cropped.shape[1]}x{cropped.shape[0]}")
        
        # Encode to JPEG and base64
        _, buffer = cv2.imencode('.jpg', cropped)
        b64 = base64.b64encode(buffer).decode('utf-8')
        
        return b64
        
    except Exception as e:
        print("❌ ERROR in crop_receipt_auto:", e)
        traceback.print_exc()
        return None


# Uses pre-calculated coordinates - NO Vision API call!
def crop_with_coordinates(image_base64, x_coords, y_coords, padding=10):
    """Crops image using pre-calculated bounding box coordinates."""
    try:
        # Decode base64 image
        img_data = base64.b64decode(image_base64)
        im = Image.open(io.BytesIO(img_data))
        
        # Check if we have coordinates
        if not x_coords or not y_coords:
            return None
        
        # Find the bounding box (smallest box that contains all text)
        left = min(x_coords)
        top = min(y_coords)
        right = max(x_coords)
        bottom = max(y_coords)
        
        # Add padding but keep within image bounds
        w, h = im.size
        left   = clamp(left - padding, 0, w)
        top    = clamp(top - padding, 0, h)
        right  = clamp(right + padding, 0, w)
        bottom = clamp(bottom + padding, 0, h)
        
        # Crop the image
        cropped = im.crop((left, top, right, bottom))
        
        # Convert RGBA to RGB if needed (for PNG with transparency)
        if cropped.mode == "RGBA":
            cropped = cropped.convert("RGB")
        
        # Save to bytes and encode as base64
        output = io.BytesIO()
        cropped.save(output, format="JPEG")
        output.seek(0)
        b64 = base64.b64encode(output.read()).decode("utf-8")
        
        return b64
        
    except Exception as e:
        print("❌ ERROR in crop_with_coordinates:", e)
        traceback.print_exc()
        return None


# ENDPOINT 1: OpenCV auto-detection (RECOMMENDED! 🔥)
@app.route("/crop-auto", methods=["POST"])
def crop_auto():
    """
    Auto-detects receipt boundaries using OpenCV.
    NO Vision API calls = FREE and FAST!
    
    Accepts either:
    1. File upload (form-data with 'file' key)
    2. JSON with 'image_base64' field
    """
    temp_input = None
    try:
        if not require_api_key():
            return jsonify({"error": "Unauthorized"}), 401

        padding = 10
        b64_result = None

        # Check if it's a file upload or JSON
        if request.content_type and 'multipart/form-data' in request.content_type:
            # File upload method
            if "file" not in request.files:
                return jsonify({"error": "No file uploaded; send as form-data with key 'file'"}), 400

            f = request.files["file"]
            if f.filename == "":
                return jsonify({"error": "Empty filename"}), 400

            allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
            _, ext = os.path.splitext(f.filename.lower())
            if ext.lstrip('.') not in allowed_extensions:
                return jsonify({"error": f"File type not supported. Allowed: {', '.join(allowed_extensions)}"}), 400

            temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            f.save(temp_input.name)

            print(f"📄 Received file: {f.filename} ({ext})")
            padding = int(request.args.get('padding', 10))
            b64_result = crop_receipt_auto(temp_input.name, padding=padding, is_base64=False)

        else:
            # JSON method
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data or file provided"}), 400
                
            if "image_base64" not in data:
                return jsonify({"error": "Missing 'image_base64' field in JSON"}), 400
            
            image_base64 = data["image_base64"]
            padding = data.get("padding", 10)
            
            print(f"📦 Processing base64 image, padding={padding}")
            b64_result = crop_receipt_auto(image_base64, padding=padding, is_base64=True)
        
        if not b64_result:
            return jsonify({"error": "Auto-crop failed"}), 422

        return jsonify({
            "status": "ok",
            "cropped_image_base64": b64_result,
            "format": "jpg",
            "method": "opencv_auto_detection"
        })

    except Exception as e:
        print("❌ SERVER ERROR:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
    finally:
        if temp_input and os.path.exists(temp_input.name):
            try:
                os.unlink(temp_input.name)
            except:
                pass


# ENDPOINT 2: Pre-calculated coordinates (if NGR already has coords)
@app.route("/crop-with-boxes", methods=["POST"])
def crop_with_boxes():
    """Accepts image + bounding boxes from NGR, NO Vision API call"""
    try:
        if not require_api_key():
            return jsonify({"error": "Unauthorized"}), 401

        data = request.get_json()
        
        # Validate required fields
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        if "image_base64" not in data:
            return jsonify({"error": "Missing 'image_base64' field"}), 400
            
        if "bounding_boxes" not in data:
            return jsonify({"error": "Missing 'bounding_boxes' field"}), 400
        
        # Extract data
        image_base64 = data["image_base64"]
        bounding_boxes = data["bounding_boxes"]
        padding = data.get("padding", 10)
        
        # Validate bounding boxes structure
        if "x_coords" not in bounding_boxes or "y_coords" not in bounding_boxes:
            return jsonify({"error": "bounding_boxes must contain 'x_coords' and 'y_coords' arrays"}), 400
        
        x_coords = bounding_boxes["x_coords"]
        y_coords = bounding_boxes["y_coords"]
        
        if not x_coords or not y_coords:
            return jsonify({"error": "x_coords and y_coords cannot be empty"}), 400
        
        print(f"📦 Processing image with {len(x_coords)} coordinates, padding={padding}")
        
        # Crop the image
        b64 = crop_with_coordinates(image_base64, x_coords, y_coords, padding)
        
        if not b64:
            return jsonify({"error": "Crop failed"}), 422

        return jsonify({
            "status": "ok",
            "cropped_image_base64": b64,
            "format": "jpg",
            "coordinates_used": len(x_coords)
        })

    except Exception as e:
        print("❌ SERVER ERROR:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "status": "ready",
        "message": "Receipt Cropping API - Zero Vision API Costs! 🎉",
        "endpoints": {
            "/crop-auto": "POST - OpenCV auto-detection (RECOMMENDED! Works with file upload or base64)",
            "/crop-with-boxes": "POST JSON - Uses pre-calculated coordinates from NGR"
        },
        "usage": {
            "file_upload": "POST /crop-auto with form-data 'file' field",
            "base64": "POST /crop-auto with JSON {'image_base64': '...', 'padding': 10}"
        },
        "note": "No Google Vision API calls = No extra costs! 💰"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)