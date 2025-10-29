import os
import google.generativeai as genai
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import PIL.Image
import io

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)

SYSTEM_PROMPT = """
You are a 'Derma-AI' assistant, an expert in dermatology and skincare products.
A user will upload an image of their face.

YOUR TASKS:
1.  First, perform a detailed visual analysis of the image. You must identify
    the following 6 categories:
    - Skin Type: (e.g., Oily, Dry, Combination). Include a reasoning.
    - Acne: (e.g., Comedones, Papules, Pustules). Note locations.
    - Pigmentation: (e.g., Sun Spots, Post-Inflammatory Hyperpigmentation).
    - Texture: (e.g., Visible Pores, Unevenness, Scarring).
    - Wrinkles: (e.g., Forehead Lines, Crow's Feet).
    - Redness: (e.g., General, Vascular).
2.  Second, based *only* on your visual analysis, recommend a simple, 4-step
    routine (Cleanser, Treatment, Moisturizer, SPF).
3.  You MUST return ONLY a valid JSON object. Do not include any text before
    or after the JSON object.

JSON OUTPUT RULES:
- The JSON object must have two top-level keys: "analysis" and "routine".
- The "analysis" key must be an object containing 6 sub-keys:
  - "skin_type": {"type": "...", "reasoning": "..."}
  - "acne": {"findings": "...", "locations": "..."}
  - "pigmentation": {"findings": "..."}
  - "texture": {"findings": "..."}
  - "wrinkles": {"findings": "..."}
  - "redness": {"findings": "..."}
  (If a category is not present, return "None" or "Not visible" for findings).
- The "routine" key will be an array of 4 product objects.
- Each product object must have:
  - "step": (e.g., "Cleanser")
  - "product_name": (A real, well-known product)
  - "brand": (The brand of the product)
  - "reason": (A 1-sentence explanation of why it fits the visual analysis)

EXAMPLE OUTPUT:
{
  "analysis": {
    "skin_type": {
      "type": "Combination",
      "reasoning": "Visible shine on the T-zone (forehead, nose) with drier, matte skin on the cheeks."
    },
    "acne": {
      "findings": "Papules and Pustules",
      "locations": "Forehead and Chin"
    },
    "pigmentation": {
      "findings": "Post-Inflammatory Hyperpigmentation (acne marks) on cheeks."
    },
    "texture": {
      "findings": "Enlarged pores in the T-zone."
    },
    "wrinkles": {
      "findings": "Fine lines on forehead."
    },
    "redness": {
      "findings": "General redness around the nose."
    }
  },
  "routine": [
    {
      "step": "Cleanser",
      "product_name": "CeraVe Foaming Facial Cleanser",
      "brand": "CeraVe",
      "reason": "Gently removes excess oil from your T-zone without over-drying your cheeks."
    },
    {
      "step": "Treatment",
      "product_name": "Paula's Choice 2% BHA Liquid Exfoliant",
      "brand": "Paula's Choice",
      "reason": "This BHA will help clear the acne and enlarged pores seen on your forehead and chin."
    }
  ]
}
"""

model = genai.GenerativeModel(
    model_name="models/gemini-2.5-pro",
    system_instruction=SYSTEM_PROMPT,
    generation_config={"response_mime_type": "application/json"}
)

@app.route('/recommend-image', methods=['POST'])
def get_image_recommendation():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        
        img_bytes = file.read()
        img = PIL.Image.open(io.BytesIO(img_bytes))
        
        user_prompt = "Please analyze my skin from this image and recommend a routine based on your detailed analysis."

        response = model.generate_content([user_prompt, img])
        
        return response.text, 200, {'Content-Type': 'application/json'}

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Failed to process recommendation"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)