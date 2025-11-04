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
Your ONLY job is to find the single best product from a given list that matches a
user's skin profile.

INPUTS:
- **profile**: A JSON object of the user's skin profile.
- **product_type**: The type of product to find (e.g., "CLEANSER").
- **products**: A JSON list of candidate products of that type, including their
  names and ingredient lists.

TASK:
1.  Analyze the user's **profile**.
2.  Examine the **ingredient list (`ingredients_str`)** of *every single product*
    in the **products** list.
3.  Find the product whose ingredients *best* target the user's specific **concerns**
    while respecting their **sensitivity**.
4.  Return ONLY a valid JSON object with the *exact name* of your chosen product
    and a *specific reasoning*.

EXAMPLE OUTPUT:
{
  "best_product_name": "CeraVe SA Cleanser",
  "reasoning": "I chose this product because its Salicylic Acid directly targets the user's 'Comedonal Acne', while its fragrance-free formula is perfect for their 'high sensitivity'."
}
"""

model_analyzer = genai.GenerativeModel(
    model_name="models/gemini-2.5-pro",
    system_instruction=ANALYZER_PROMPT,
    generation_config={"response_mime_type": "application/json"}
)

model_recommender = genai.GenerativeModel(
    model_name="models/gemini-2.5-pro",
    system_instruction=RECOMMENDER_PROMPT,
    generation_config={"response_mime_type": "application/json"}
)

def get_recommendation_for_type(skin_profile, product_type):
    if product_db.empty:
        return None
    
    candidate_df = product_db[product_db['type'] == product_type]
    if candidate_df.empty:
        return None 
    
    sample_size = min(len(candidate_df), 75)
    product_sample = candidate_df.sample(sample_size)
    
    products_json = product_sample[['name', 'ingredients_str']].to_dict(orient='records')
    
    prompt = f"""
    "profile": {json.dumps(skin_profile)},
    "product_type": "{product_type}",
    "products": {json.dumps(products_json)}
    """
    
    try:
        response = model_recommender.generate_content(prompt)
        recommendation = json.loads(response.text)
        
        best_name = recommendation.get("best_product_name")
        if not best_name:
            return None
        
        full_product_data = product_sample[product_sample['name'].str.lower() == best_name.lower()]
        
        if not full_product_data.empty:
            return full_product_data.iloc[0].to_dict()
        else:
            return product_sample.sample(1).iloc[0].to_dict()

    except Exception as e:
        print(f"Error in recommender AI call: {e}")
        return product_sample.sample(1).iloc[0].to_dict()

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

        recommended_products = []
        product_types_to_find = ['CLEANSER', 'TONER', 'SERUM', 'MOISTURIZER', 'SUNSCREEN']
        
        for p_type in product_types_to_find:
            print(f"Calling Recommender AI for: {p_type}...")
            product = get_recommendation_for_type(skin_profile, p_type)
            
            if product:
                recommended_products.append({
                    "imageUrl": product.get('imageUrl'),
                    "type": product.get('type'),
                    "price": product.get('price'),
                    "name": product.get('name'),
                    "store": product.get('store')
                })

        return jsonify(recommended_products), 200

    except json.JSONDecodeError:
        print(f"Error: Invalid JSON string provided in 'data' field.")
        return jsonify({"error": "Invalid JSON string in 'data' field."}), 400
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Failed to process recommendation"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)