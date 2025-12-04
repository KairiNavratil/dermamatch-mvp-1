import os
import logging
import sys
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import PIL.Image
import io
import json
import pandas as pd
import random
from face_mapper import FaceMapper

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

load_dotenv()

# Configure Gemini (Only fails if you try to use the Real endpoints without a key)
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    logger.warning("⚠️ GEMINI_API_KEY not found. Real AI endpoints will fail.")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

mapper = FaceMapper()

# --- MOCK DATA FOR TEST ENDPOINTS ---
MOCK_ANALYSIS_RESPONSE = {
    "analysis": [
        {
            "x": 45.5, 
            "y": 40.2, 
            "title": "[TEST] Forehead Texture", 
            "description": "Simulated roughness detected (Test Mode).", 
            "anchor_used": "mock_id_1"
        },
        {
            "x": 60.1, 
            "y": 65.8, 
            "title": "[TEST] Chin Redness", 
            "description": "Simulated inflammation detected (Test Mode).", 
            "anchor_used": "mock_id_2"
        }
    ],
    # Added dummy landmarks so "Debug Mesh" works in frontend during testing
    "all_landmarks": [
        {"code": "mock_id_1", "x": 45.5, "y": 40.2, "region": "Forehead"},
        {"code": "mock_id_2", "x": 60.1, "y": 65.8, "region": "Chin"}
    ]
}

MOCK_ROUTINE_RESPONSE = {
    "routine": [
        {
            "step": "Cleanser",
            "product_name": "[TEST] Gentle Foaming Wash",
            "brand": "TestBrand",
            "price": 12.99,
            "store": "TestStore",
            "image_url": "https://via.placeholder.com/150",
            "benefits": [{"title": "Gentle", "description": "Good for testing."}]
        },
        {
            "step": "Treatment",
            "product_name": "[TEST] Salicylic Acid Serum",
            "brand": "The Test Ordinary",
            "price": 8.50,
            "store": "Sephora",
            "image_url": "https://via.placeholder.com/150",
            "benefits": [{"title": "Exfoliating", "description": "Clears mock acne."}]
        },
        {
            "step": "Moisturizer",
            "product_name": "[TEST] Hydra-Gel",
            "brand": "Neutro-Test",
            "price": 18.00,
            "store": "Target",
            "image_url": "https://via.placeholder.com/150",
            "benefits": [{"title": "Hydrating", "description": "Water-based hydration."}]
        },
        {
            "step": "SPF",
            "product_name": "[TEST] Invisible Sunscreen",
            "brand": "Super-Test",
            "price": 30.00,
            "store": "Ulta",
            "image_url": "https://via.placeholder.com/150",
            "benefits": [{"title": "Protection", "description": "SPF 50 mock protection."}]
        }
    ]
}

# --- DATABASE LOADING ---
product_db = pd.DataFrame()
try:
    if os.path.exists("cleaned_products.csv"):
        logger.info("Loading product database...")
        product_db = pd.read_csv("cleaned_products.csv")
        
        cols_to_str = ['name', 'brand', 'ingredients_str', 'type']
        for col in cols_to_str:
            if col in product_db.columns:
                product_db[col] = product_db[col].fillna('').astype(str)
            else:
                product_db[col] = '' 

        if 'price' in product_db.columns:
            product_db['price'] = product_db['price'].fillna(0)
        
        if 'store' in product_db.columns:
            product_db['store'] = product_db['store'].fillna('')

        product_db['type'] = product_db['type'].str.upper().str.strip()
        logger.info(f"✅ Loaded Product DB: {len(product_db)} items.")
    else:
        logger.warning("⚠️ WARNING: cleaned_products.csv not found.")
except Exception as e:
    logger.error(f"❌ Error loading DB: {e}")

def get_product_candidates(product_type, limit=50):
    if product_db.empty: return []
    candidates = product_db[product_db['type'] == product_type.upper()]
    if candidates.empty: return []
    sample_size = min(len(candidates), limit)
    sample = candidates.sample(sample_size)
    return sample[['name', 'brand', 'ingredients_str', 'imageUrl', 'price', 'store']].to_dict(orient='records')

# --- PROMPTS ---
ANALYSIS_PROMPT = """
You are a 'Derma-AI' specialist. 
Your goal is to analyze a selfie for skin features and map them to the PRECISELY matching point from the provided mesh.

INPUT DATA:
1. An image of a face.
2. A complete list of 468 Facial Landmarks ("Mesh Points").
   - Each point has an ID, a Region Name, and **X/Y Coordinates (0-100%)**.

YOUR TASK:
1. **Visual Scan:** Identify skin features (acne, texture, redness) on the image.
2. **Triangulate:**
   - Estimate the X/Y position of the feature you see (e.g., "This pimple is roughly at 50% width, 60% height").
   - Scan the provided list of Mesh Points.
   - Find the ID whose X/Y coordinates are MATHEMATICALLY CLOSEST to your estimated feature location.
   - Use the 'Region' name only as a sanity check.

CRITICAL RULES:
- **Precision:** Do not just pick a random point in the region. Compare the coordinates!
- **Output:** JSON Array only.

OUTPUT FORMAT:
[
  {
    "anchor_code": "id_152",
    "title": "Chin Papule",
    "description": "Inflamed spot on the chin center."
  }
]
"""

ROUTINE_PROMPT = """
You are an elite Cosmetic Chemist AI. Your goal is to curate a highly personalized routine.

INPUT DATA:
1. User Image (Visual context).
2. Survey Answers (Habits, Sensitivity).
3. **Product Candidates**: List of available products.

YOUR TASK:
1. **Analyze the User:** Combine visual needs + survey data.
2. **Select Products:** For each step (Cleanser, Treatment, Moisturizer, SPF), pick the BEST match based on ingredients.
3. **ESTIMATE MISSING DATA:** If 'price' is 0 or 'store' is empty, ESTIMATE them based on the brand's typical market positioning.

OUTPUT FORMAT (JSON):
{
  "routine": [
    {
      "step": "Cleanser",
      "product_name": "Name",
      "brand": "Brand",
      "price": "Number",
      "store": "String",
      "image_url": "URL",
      "benefits": [ { "title": "Benefit", "description": "..." } ]
    }
  ]
}
"""

# --- INITIALIZE MODELS ---
try:
    visual_model = genai.GenerativeModel(
        model_name="gemini-2.5-flash-preview-09-2025", 
        system_instruction=ANALYSIS_PROMPT,
        generation_config={"response_mime_type": "application/json"}
    )

    routine_model = genai.GenerativeModel(
        model_name="gemini-2.5-flash-preview-09-2025", 
        system_instruction=ROUTINE_PROMPT,
        generation_config={"response_mime_type": "application/json"}
    )
except Exception as e:
    logger.error(f"Error initializing Gemini models: {e}")

# --- HEALTH CHECK ---
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "online", 
        "endpoints": [
            "/recommend-image (REAL)",
            "/recommend-image-test (MOCK)",
            "/recommend-routine (REAL)",
            "/recommend-routine-test (MOCK)"
        ]
    }), 200


# ==========================================
#  ENDPOINT 1: VISUAL ANALYSIS
# ==========================================

@app.route('/recommend-image', methods=['POST'])
def get_image_recommendation_real():
    """REAL: Uses MediaPipe + Gemini (Costs Tokens)"""
    try:
        logger.info("--- Starting /recommend-image (REAL) ---")
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        img_bytes = file.read()
        
        logger.info("Running MediaPipe...")
        face_data = mapper.get_face_landmarks(img_bytes)
        if not face_data:
            return jsonify({"error": "No face detected in image."}), 400

        ai_landmark_context = []
        all_landmarks_debug = []
        
        for code, data in face_data['landmarks'].items():
            x_pct = (data['x'] / face_data['width']) * 100
            y_pct = (data['y'] / face_data['height']) * 100
            
            all_landmarks_debug.append({
                "code": code,
                "x": x_pct,
                "y": y_pct,
                "region": data['region']
            })
            
            ai_string = f"ID: {code} | Region: {data['region']} | Location: X={x_pct:.1f}%, Y={y_pct:.1f}%"
            ai_landmark_context.append(ai_string)

        img_pil = PIL.Image.open(io.BytesIO(img_bytes))
        
        user_prompt = f"""
        Analyze this image. 
        Here is the MAP of the face with exact coordinates:
        {json.dumps(ai_landmark_context)}
        
        INSTRUCTION: 
        1. Find a blemish visually.
        2. Estimate its X/Y coordinates on the image.
        3. Find the ID in the list above that has the CLOSEST coordinates.
        """

        logger.info("Sending request to Gemini...")
        response = visual_model.generate_content([user_prompt, img_pil])
        ai_results = json.loads(response.text)
        
        final_response = []
        for item in ai_results:
            code = item.get('anchor_code')
            if code in face_data['landmarks']:
                coords = face_data['landmarks'][code]
                x_pct = (coords['x'] / face_data['width']) * 100
                y_pct = (coords['y'] / face_data['height']) * 100
                final_response.append({
                    "x": x_pct,
                    "y": y_pct,
                    "title": item.get('title'),
                    "description": item.get('description'),
                    "anchor_used": code
                })
        
        return jsonify({
            "analysis": final_response,
            "all_landmarks": all_landmarks_debug
        }), 200

    except Exception as e:
        logger.exception("Error in /recommend-image")
        return jsonify({"error": str(e)}), 500


@app.route('/recommend-image-test', methods=['POST'])
def get_image_recommendation_test():
    """TEST: Returns static mock data (No Cost)"""
    logger.info("--- Starting /recommend-image-test (MOCK) ---")
    return jsonify(MOCK_ANALYSIS_RESPONSE), 200


# ==========================================
#  ENDPOINT 2: ROUTINE GENERATION
# ==========================================

@app.route('/recommend-routine', methods=['POST'])
def get_routine_recommendation_real():
    """REAL: Uses Gemini to generate routine (Costs Tokens)"""
    try:
        logger.info("--- Starting /recommend-routine (REAL) ---")
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        img_bytes = file.read()
        img_pil = PIL.Image.open(io.BytesIO(img_bytes))

        survey_data_str = request.form.get('data')
        if not survey_data_str:
            return jsonify({"error": "No survey data provided"}), 400
        survey_data = json.loads(survey_data_str)

        logger.info("Fetching candidates...")
        candidates = {
            "Cleanser": get_product_candidates("CLEANSER", 50),
            "Treatment": get_product_candidates("SERUM", 50), 
            "Moisturizer": get_product_candidates("MOISTURIZER", 50),
            "SPF": get_product_candidates("SUNSCREEN", 50)
        }

        user_prompt = f"""
        Generate a skincare routine.
        SURVEY: {json.dumps(survey_data, indent=2)}
        CANDIDATES: {json.dumps(candidates, indent=2)}
        """

        logger.info("Sending request to Gemini...")
        response = routine_model.generate_content([user_prompt, img_pil])
        routine_results = json.loads(response.text)
        
        return jsonify(routine_results), 200

    except Exception as e:
        logger.exception("Error in /recommend-routine")
        return jsonify({"error": str(e)}), 500


@app.route('/recommend-routine-test', methods=['POST'])
def get_routine_recommendation_test():
    """TEST: Returns static mock data (No Cost)"""
    logger.info("--- Starting /recommend-routine-test (MOCK) ---")
    return jsonify(MOCK_ROUTINE_RESPONSE), 200


if __name__ == '__main__':
    app.run(debug=True, port=5000)