"""
Skin Disease Detector - Python Flask Backend (Google Gemini 2026)
=============================================================
BE Engineering Project Demo — HAM10000 Dataset (7 classes)

Setup:
    1. pip uninstall google-generativeai
    2. pip install google-genai flask flask-cors pillow python-dotenv pydantic
    3. Open .env and paste your Gemini API key (GEMINI_API_KEY=AIza...)
    4. python app.py
    5. Right-click index.html → Open with Live Server
"""

import os
import base64
import json
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from dotenv import load_dotenv

# ✅ Use the modern Google GenAI SDK
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()

app = Flask(__name__)
CORS(app)

# ✅ Initialize the modern client (it automatically looks for GEMINI_API_KEY env var)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    # Fallback check if it's named slightly differently in your .env
    GEMINI_API_KEY = os.environ.get("GENAI_API_KEY", "")

client = genai.Client(api_key=GEMINI_API_KEY)

DISEASES = {
    "akiec": "Actinic Keratoses & Intraepithelial Carcinoma",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis-like Lesions",
    "df":    "Dermatofibroma",
    "nv":    "Melanocytic Nevi",
    "mel":   "Melanoma",
    "vasc":  "Vascular Lesions",
}

DISEASE_LIST = "\n".join([f'- "{k}": {v}' for k, v in DISEASES.items()])


# ✅ Define the strict JSON structure using Pydantic
# This guarantees confidence_score is a strict float value.
class DiseaseAnalysis(BaseModel):
    code: str = Field(description="Short code like akiec, bcc, or 'unknown' if not a lesion")
    condition: str = Field(description="Full condition name")
    confidence_label: str = Field(description="High / Medium / Low")
    confidence_score: float = Field(description="Float certainty score strictly between 0.0 and 1.0")
    description: str = Field(description="2-sentence plain language description of the condition")
    symptoms: list[str] = Field(description="Array of exactly 3 symptoms")
    commonIn: str = Field(description="Who is commonly affected")
    riskLevel: str = Field(description="Low / Moderate / High / Critical")
    action: str = Field(description="Recommended next step in one sentence")


PROMPT = f"""You are a dermatology AI assistant for a BE engineering project demo using the HAM10000 dataset.
Analyze the skin lesion image and classify it into one of these 7 categories:

{DISEASE_LIST}

If the image does not show a skin lesion, set code to 'unknown', condition to 'No skin lesion detected', confidence_label to 'Low', and confidence_score to 0.0.
IMPORTANT: For educational/demo purposes only. Always recommend consulting a dermatologist."""

MAX_IMAGE_SIZE = (1024, 1024)
MAX_UPLOAD_MB  = 10


def prepare_image(file_bytes: bytes):
    img = Image.open(io.BytesIO(file_bytes))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.thumbnail(MAX_IMAGE_SIZE, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return buf.read()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "gemini-2.5-flash", "diseases": list(DISEASES.keys())})


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image provided. Send field name 'image'."}), 400

    file = request.files["image"]
    file_bytes = file.read()

    if len(file_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        return jsonify({"error": f"File too large. Max {MAX_UPLOAD_MB} MB."}), 413

    try:
        img_bytes = prepare_image(file_bytes)
    except Exception as e:
        return jsonify({"error": f"Could not process image: {str(e)}"}), 400

    try:
        # ✅ Modern SDK structural configuration
        response = client.models.generate_content(
            model='gemini-2.5-flash', # Using stable 2.5 flash
            contents=[
                types.Part.from_bytes(
                    data=img_bytes,
                    mime_type='image/jpeg',
                ),
                PROMPT
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DiseaseAnalysis, # Enforces structural parsing
                temperature=0.1 # Low temperature keeps confidence scoring grounded
            ),
        )
        raw_text = response.text

    except Exception as e:
        err = str(e)
        if "API_KEY" in err or "api key" in err.lower():
            return jsonify({"error": "Invalid Gemini API key. Check GEMINI_API_KEY in your .env file."}), 500
        return jsonify({"error": f"Gemini API error: {err}"}), 500

    try:
        # ✅ Because of response_schema + Pydantic, this is guaranteed to be clean JSON
        result = json.loads(raw_text)
        
        # Double check/clamp value constraints
        if "confidence_score" in result:
            result["confidence_score"] = max(0.0, min(1.0, float(result["confidence_score"])))

    except (json.JSONDecodeError, ValueError):
        return jsonify({"error": "Model failed structural validation.", "raw": raw_text[:300]}), 500

    return jsonify(result)


if __name__ == "__main__":
    if not GEMINI_API_KEY:
        print("\n⚠️  GEMINI_API_KEY not set!")
        print("   Open .env and paste your key:  GEMINI_API_KEY=AIza...\n")
    else:
        print(f"✅ Gemini API key loaded (...{GEMINI_API_KEY[-4:]})")

    print("\n🔬 Skin Disease Detector — HAM10000 (7 classes) — New Gemini SDK")
    print("   Backend  →  http://localhost:5000")
    print("   Frontend →  Right-click index.html → Open with Live Server\n")

    app.run(debug=True, host="0.0.0.0", port=5000)