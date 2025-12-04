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
import gc 
import time
from face_mapper import FaceMapper
from google.api_core import exceptions

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

load_dotenv()

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    logger.warning("⚠️ GEMINI_API_KEY not found. Real AI endpoints will fail.")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# NOTE: We do NOT initialize mapper = FaceMapper() here anymore.
# We initialize it inside the route to save memory on Render.

# --- MOCK DATA ---
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
    "all_landmarks": [
        {"code": "mock_id_1", "x": 45.5, "y": 40.2, "region": "Forehead"},
        {"code": "mock_id_2", "x": 60.1, "y": 65.8, "region": "Chin"}
    ]
}

MOCK_ROUTINE_RESPONSE = {
    "routine": [
        {
            "imageUrl": "https://via.placeholder.com/150",
            "name": "[TEST] Gentle Foaming Wash",
            "brand": "TestBrand",
            "price": 12.99,
            "store": "TestStore",
            "type": "CLEANSER",
            "benefits": [{"title": "Why this?", "description": "Gentle foam removes oil without stripping."}]
        },
        {
            "imageUrl": "https://via.placeholder.com/150",
            "name": "[TEST] Balancing Toner",
            "brand": "TestBrand",
            "price": 14.50,
            "store": "Ulta",
            "type": "TONER",
            "benefits": [{"title": "Why this?", "description": "Restores pH balance after cleansing."}]
        },
        {
            "imageUrl": "https://via.placeholder.com/150",
            "name": "[TEST] Salicylic Acid Serum",
            "brand": "The Test Ordinary",
            "price": 8.50,
            "store": "Sephora",
            "type": "SERUM",
            "benefits": [{"title": "Why this?", "description": "Targets the acne detected on the chin."}]
        },
        {
            "imageUrl": "https://via.placeholder.com/150",
            "name": "[TEST] Hydra-Gel",
            "brand": "Neutro-Test",
            "price": 18.00,
            "store": "Target",
            "type": "MOISTURIZER",
            "benefits": [{"title": "Why this?", "description": "Lightweight hydration for oily skin."}]
        },
        {
            "imageUrl": "https://via.placeholder.com/150",
            "name": "[TEST] Invisible Sunscreen",
            "brand": "Super-Test",
            "price": 30.00,
            "store": "Ulta",
            "type": "SUNSCREEN",
            "benefits": [{"title": "Why this?", "description": "Protects sensitive skin without whitecast."}]
        }
    ]
}

# --- DATABASE LOADING ---
product_db = pd.DataFrame()
INVENTORY_CONTEXT = "" # Global string to hold the entire catalog

try:
    if os.path.exists("cleaned_products.csv"):
        logger.info("Loading product database...")
        df_raw = pd.read_csv("cleaned_products.csv")
        
        # 1. Normalize Columns
        cols_to_str = ['name', 'brand', 'ingredients_str', 'type']
        for col in cols_to_str:
            if col in df_raw.columns:
                df_raw[col] = df_raw[col].fillna('').astype(str)
            else:
                df_raw[col] = '' 

        if 'price' in df_raw.columns:
            df_raw['price'] = df_raw['price'].fillna(0)
        
        if 'store' in df_raw.columns:
            df_raw['store'] = df_raw['store'].fillna('')
        
        # Ensure imageUrl is clean string
        if 'imageUrl' not in df_raw.columns:
             df_raw['imageUrl'] = ''
        
        # Force string, fill NA, and STRIP WHITESPACE
        df_raw['imageUrl'] = df_raw['imageUrl'].fillna('').astype(str).str.strip()
        df_raw['type'] = df_raw['type'].str.upper().str.strip()

        # 2. FILTER: Only keep the core 5 steps.
        target_types = ['CLEANSER', 'TONER', 'SERUM', 'MOISTURIZER', 'SUNSCREEN']
        
        valid_mask = (df_raw['type'].isin(target_types))
        df_clean = df_raw[valid_mask].copy()
        
        # Fallback if filtering fails
        if len(df_clean) < 50:
            logger.warning("⚠️ Warning: Filtering removed almost all products! Falling back to full DB.")
            product_db = df_raw
        else:
            logger.info(f"✅ Filtered to {len(df_clean)} products (Core Categories Only).")
            product_db = df_clean

        # 3. GENERATE COMPRESSED CONTEXT (ID|Type|Brand|Name)
        # We assume the dataframe index is the stable ID
        product_db['context_str'] = (
            "[" + product_db.index.astype(str) + "]" + 
            product_db['type'] + "|" + 
            product_db['brand'] + "|" + 
            product_db['name']
        )
        
        INVENTORY_CONTEXT = "\n".join(product_db['context_str'].tolist())
        
        logger.info(f"✅ Inventory Context Ready ({len(INVENTORY_CONTEXT)} chars)")
        
    else:
        logger.warning("⚠️ WARNING: cleaned_products.csv not found.")
except Exception as e:
    logger.error(f"❌ Error loading DB: {e}")


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
- **No Location Text:** Do NOT describe the location (e.g. "left cheek", "at 50% height") in the title or description. The UI marker shows the location. Focus strictly on WHAT the issue is (e.g. "Inflamed Papule", "Dry Patch").
- **Output:** JSON Array only.

OUTPUT FORMAT:
[
  {
    "anchor_code": "id_152",
    "title": "Inflamed Papule",
    "description": "Redness and slight swelling indicating active inflammation."
  }
]
"""

# UPDATED: Ask for estimated prices/stores
ROUTINE_PROMPT = """
You are an elite Cosmetic Chemist AI. Curate a routine by SELECTING items from the inventory.

INPUT:
1. User Image (Visual context).
2. Survey Answers.
3. **INVENTORY CATALOG:** `[ID]TYPE|BRAND|NAME`

TASK:
1. Analyze user needs.
2. Scan CATALOG. Pick the BEST product ID for: Cleanser, Toner, Serum, Moisturizer, Sunscreen.
3. **Data Filling:** Estimate the market price (USD) and a common US retailer (e.g. Sephora, Target, Amazon) for each selected item.
4. **Personalize:** Write a VERY SHORT (1 sentence) reason why it fits this **skin profile** (avoid addressing the user directly as 'you').

OUTPUT FORMAT (JSON):
{
  "routine": [
    { 
      "step": "Cleanser", 
      "inventory_id": 123, 
      "estimated_price": 15.00,
      "estimated_store": "Target",
      "short_reason": "Contains salicylic acid to target chin acne." 
    },
    ...
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

# --- HELPER: RETRY LOGIC ---
def generate_with_retry(model, inputs, retries=3):
    """Retries generation if 429 (Resource Exhausted) occurs."""
    for attempt in range(retries):
        try:
            return model.generate_content(inputs)
        except exceptions.ResourceExhausted:
            wait_time = (2 ** attempt) + 1  # Exponential backoff: 2s, 3s, 5s...
            logger.warning(f"⚠️ Quota exceeded (429). Retrying in {wait_time}s...")
            time.sleep(wait_time)
        except Exception as e:
            # For other errors, raise immediately
            raise e
    raise Exception("Max retries exceeded for AI generation.")

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
    local_mapper = None
    try:
        logger.info("--- Starting /recommend-image (REAL) ---")
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        
        # --- MEMORY OPTIMIZATION STEP ---
        image = PIL.Image.open(file)
        
        max_size = 1024
        if image.width > max_size or image.height > max_size:
            logger.info(f"Resizing image from {image.size} to max {max_size}px...")
            image.thumbnail((max_size, max_size))
            
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format or 'JPEG')
        img_bytes = img_byte_arr.getvalue()
        
        # --- LAZY LOADING MEDIAPIPE ---
        logger.info("Initializing FaceMapper locally...")
        local_mapper = FaceMapper()
        
        logger.info("Running MediaPipe...")
        face_data = local_mapper.get_face_landmarks(img_bytes)
        
        # Force cleanup
        local_mapper.face_mesh.close()
        del local_mapper
        local_mapper = None
        gc.collect() 
        
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
        # Use retry logic here too
        response = generate_with_retry(visual_model, [user_prompt, image])
        logger.info("Gemini response received.")
        
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
        if local_mapper:
            try:
                local_mapper.face_mesh.close()
            except:
                pass
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
    """REAL: Uses Gemini with FULL DATABASE CONTEXT"""
    try:
        logger.info("--- Starting /recommend-routine (REAL) ---")
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        image = PIL.Image.open(file)
        image.thumbnail((1024, 1024))
        
        survey_data_str = request.form.get('data')
        if not survey_data_str:
            return jsonify({"error": "No survey data provided"}), 400
        survey_data = json.loads(survey_data_str)

        # 1. CONSTRUCT PROMPT WITH COMPRESSED INVENTORY
        user_prompt = f"""
        Generate a skincare routine for this user.
        
        USER SURVEY: {json.dumps(survey_data, indent=2)}
        
        === INVENTORY CATALOG START ===
        {INVENTORY_CONTEXT}
        === INVENTORY CATALOG END ===
        
        INSTRUCTIONS:
        - Pick exactly one product for: Cleanser, Toner, Serum, Moisturizer, Sunscreen.
        - You MUST pick by ID from the list above.
        """

        logger.info("Asking Gemini to select from Inventory...")
        
        # Use retry logic for robustness
        response = generate_with_retry(routine_model, [user_prompt, image])
        ai_selection = json.loads(response.text)
        
        # 2. CONVERT IDs TO FULL PRODUCT DETAILS
        final_routine = []
        
        for step in ai_selection.get("routine", []):
            inv_id = step.get('inventory_id')
            
            # Lookup in DB
            if inv_id is not None and inv_id in product_db.index:
                row = product_db.loc[inv_id]
                logger.info(f"✅ Selected ID {inv_id}: {row['name']}")
                
                # Retrieve URL and clean it
                url_val = row.get('imageUrl', '').strip()
                
                # AI Generated Benefit & Estimates
                ai_reason = step.get('short_reason', 'Selected based on your skin analysis.')
                ai_price = step.get('estimated_price', 0)
                ai_store = step.get('estimated_store', 'Check Retailers')

                # Smart Logic: Use DB value if exists, else fallback to AI estimate
                db_price = row.get('price', 0)
                final_price = db_price if db_price > 0 else ai_price
                
                db_store = str(row.get('store', '')).strip()
                final_store = db_store if db_store not in ['', 'N/A', 'nan'] else ai_store

                # Construct Response
                final_routine.append({
                    "imageUrl": url_val,
                    "name": row['name'],
                    "price": final_price,
                    "store": final_store,
                    "type": row['type'], 
                    "brand": row['brand'],
                    "benefits": [{"title": "Why this?", "description": ai_reason}] 
                })
            else:
                logger.warning(f"⚠️ AI returned invalid ID: {inv_id}")

        return jsonify({"routine": final_routine}), 200

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