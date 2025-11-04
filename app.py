import os
import google.generativeai as genai
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import PIL.Image
import io
import json
import pandas as pd
import random

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)

try:
    product_db = pd.read_csv("cleaned_products.csv")
    product_db['name'] = product_db['name'].fillna('').astype(str)
    product_db['ingredients_str'] = product_db['ingredients_str'].fillna('').astype(str)
    print(f"Successfully loaded cleaned_products.csv. {len(product_db)} products ready.")
except FileNotFoundError:
    print("FATAL ERROR: cleaned_products.csv not found.")
    product_db = pd.DataFrame()

ANALYZER_PROMPT = """
You are a 'Derma-AI' visual analyzer. You are an expert dermatologist.
Your ONLY job is to analyze a user's image and questionnaire data and
return a JSON object describing their skin profile.

INPUTS:
- Image: A picture of the user's face.
- question1: Sensitivity score (1-5, 5=very sensitive).
- question2: Makeup frequency ("Daily", "A few times a week", "Rarely, or never").

ANALYSIS RULES:
1.  **Visual Analysis:** From the image, identify Skin Type (Oily, Dry, Combo),
    and a list of specific visual concerns (e.g., "comedonal acne", "pustular acne",
    "hyperpigmentation", "dullness", "fine lines on forehead", "redness on cheeks").
    Be as specific as possible.
2.  **Data Analysis:** Integrate the questionnaire data.

OUTPUT:
Return ONLY a valid JSON object in this exact format. Do not add any other text.

EXAMPLE OUTPUT:
{
  "skin_type": "Combination",
  "concerns": ["Comedonal Acne", "Hyperpigmentation", "Oily T-Zone", "Dehydrated Cheeks"],
  "sensitivity": 5,
  "makeup_use": "Daily"
}
"""

RECOMMENDER_PROMPT = """
You are a 'Derma-AI' product recommender. You are an expert cosmetic chemist.
Your ONLY job is to find the single best product from *each* category list
that matches a user's skin profile.

INPUTS:
- **profile**: A JSON object of the user's skin profile.
- **product_lists**: A JSON object containing 5 arrays of products.
  (e.g., "CLEANSER": [...], "TONER": [...], etc.)

TASK:
1.  Analyze the user's **profile**.
2.  For **each** of the 5 product lists provided (CLEANSER, TONER, etc.):
3.  Examine the **ingredient list (`ingredients_str`)** of every product in that list.
4.  Find the **one** product from that specific list whose ingredients *best*
    target the user's **concerns** while respecting their **sensitivity**.
5.  Based on the product's name and brand, provide a *realistic estimated price*
    (as a number, e.g., 24.99) and a *common store* (e.g., "Sephora", "Target").
6.  Return ONLY a valid JSON object containing an array named "recommendations".
    Each item in the array must have all 5 of these keys.

EXAMPLE OUTPUT:
{
  "recommendations": [
    {
      "type": "CLEANSER",
      "best_product_name": "CeraVe SA Cleanser",
      "reasoning": "Selected for its Salicylic Acid to target 'Comedonal Acne', while being fragrance-free for 'high sensitivity'.",
      "price": 14.99,
      "store": "Target"
    },
    {
      "type": "TONER",
      "best_product_name": "Anua Heartleaf 77% Soothing Toner",
      "reasoning": "The Houttuynia Cordata is excellent for calming the redness and irritation seen in the profile.",
      "price": 22.50,
      "store": "Amazon"
    }
  ]
}
"""

model_analyzer = genai.GenerativeModel(
    model_name="models/gemini-2.5-pro",
    system_instruction=ANALYZER_PROMPT,
    generation_config={"response_mime_type": "application/json"}
)

model_recommender = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash-lite",
    system_instruction=RECOMMENDER_PROMPT,
    generation_config={"response_mime_type": "application/json"}
)

def get_product_by_name(name):
    if product_db.empty or name is None:
        return None
    
    result = product_db[product_db['name'].str.lower() == name.lower()]
    
    if not result.empty:
        return result.iloc[0].to_dict()
    return None

def get_product_sample(product_type, sample_size=10):
    if product_db.empty:
        return []
    
    candidate_df = product_db[product_db['type'] == product_type]
    if candidate_df.empty:
        return []
    
    actual_sample_size = min(len(candidate_df), sample_size)
    product_sample = candidate_df.sample(actual_sample_size)
    
    return product_sample[['name', 'ingredients_str']].to_dict(orient='records')


@app.route('/recommend-image', methods=['POST'])
def get_image_recommendation():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        file = request.files['image']
        img = PIL.Image.open(io.BytesIO(file.read()))

        if 'data' not in request.form:
            return jsonify({"error": "No JSON data provided"}), 400
        data_str = request.form['data']
        user_data = json.loads(data_str)
        
        sensitivity = user_data.get('question1', 1)
        makeup = user_data.get('question2', "Rarely, or never")

        analyzer_user_prompt = f"""
        Analyze the attached image and the following user data:
        - question1 (sensitivity 1-5): {sensitivity}
        - question2 (makeup use): "{makeup}"
        """

        print("Calling Analyzer AI...")
        response = model_analyzer.generate_content([analyzer_user_prompt, img])
        skin_profile = json.loads(response.text)
        print(f"Analyzer AI returned profile: {skin_profile}")

        print("Building product lists for Recommender AI...")
        product_lists_for_ai = {
            "CLEANSER": get_product_sample("CLEANSER", 10),
            "TONER": get_product_sample("TONER", 10),
            "SERUM": get_product_sample("SERUM", 10),
            "MOISTURIZER": get_product_sample("MOISTURIZER", 10),
            "SUNSCREEN": get_product_sample("SUNSCREEN", 10)
        }

        recommender_prompt = f"""
        "profile": {json.dumps(skin_profile)},
        "product_lists": {json.dumps(product_lists_for_ai)}
        """

        print("Calling Recommender AI (Mega-Call)...")
        response = model_recommender.generate_content(recommender_prompt)
        recommendations = json.loads(response.text).get("recommendations", [])
        print(f"Recommender AI returned {len(recommendations)} products.")

        final_product_list = []
        for rec in recommendations:
            product_name = rec.get("best_product_name")
            product_data = get_product_by_name(product_name)
            
            product_image_url = "https://example.com/images/default.jpg"
            if product_data and product_data.get('imageUrl'):
                product_image_url = product_data.get('imageUrl')

            final_product_list.append({
                "imageUrl": product_image_url,
                "type": rec.get("type"),
                "price": rec.get("price", 0.0),
                "name": product_name,
                "store": rec.get("store", "N/A")
            })

        return jsonify(final_product_list), 200

    except json.JSONDecodeError:
        print(f"Error: Invalid JSON string provided in 'data' field.")
        return jsonify({"error": "Invalid JSON string in 'data' field."}), 400
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Failed to process recommendation"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)