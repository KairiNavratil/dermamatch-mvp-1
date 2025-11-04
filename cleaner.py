import pandas as pd
import json
import numpy as np

# --- Load Dataset ---
print("Loading INCIproducts.json...")
try:
    df_inci = pd.read_json("INCIproducts.json")
except Exception as e:
    print(f"Loading INCIproducts.json with lines=True due to error: {e}")
    df_inci = pd.read_json("INCIproducts.json", lines=True)

print("Finished loading INCIproducts.json.")

# --- Use INCI as the main dataframe ---
df_combined = df_inci 

# --- Clean Product Type ('type') Column ---
print("Cleaning 'type' column...")
df_combined['type_lower'] = df_combined['type'].astype(str).str.lower()
df_combined['name_lower'] = df_combined['name'].astype(str).str.lower()

def clean_type(row):
    type_low = row['type_lower']
    name_low = row['name_lower']
    
    # Guess from the 'name' since 'type' is often 'Unknown'
    if 'sunscreen' in name_low or 'spf' in name_low:
        return 'SUNSCREEN'
    if 'moisturizer' in name_low or 'cream' in name_low or 'lotion' in name_low or 'gel' in name_low:
        return 'MOISTURIZER'
    if 'serum' in name_low or 'essence' in name_low or 'ampoule' in name_low:
        return 'SERUM'
    if 'toner' in name_low:
        return 'TONER'
    if 'cleanser' in name_low or 'wash' in name_low or 'foaming' in name_low:
        return 'CLEANSER'
    
    # Check the original 'type' column as a fallback
    if 'cleanser' in type_low:
        return 'CLEANSER'
    if 'sunscreen' in type_low or 'spf' in type_low:
        return 'SUNSCREEN'

    return 'UNKNOWN'

df_combined['type'] = df_combined.apply(clean_type, axis=1)

# --- Add Missing Columns ('price', 'store', 'imageUrl') ---
print("Adding missing columns...")
df_combined.rename(columns={'image': 'imageUrl'}, inplace=True)
df_combined['price'] = 0.0  
df_combined['store'] = 'N/A' 

# --- Final Cleanup ---
print("Finalizing dataframe...")
final_columns = ['name', 'brand', 'imageUrl', 'type', 'price', 'store', 'ingredients', 'skin_types']
for col in final_columns:
    if col not in df_combined.columns:
        df_combined[col] = 'N/A' 

df_final = df_combined[final_columns]
df_final = df_final[df_final['type'] != 'UNKNOWN']
df_final['imageUrl'] = df_final['imageUrl'].fillna('https://example.com/images/default.jpg')
df_final['ingredients_str'] = df_final['ingredients'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x)).str.lower()


# --- Save Cleaned Data ---
output_filename = 'cleaned_products.csv'
print(f"Saving cleaned data to {output_filename}...")
df_final.to_csv(output_filename, index=False)

print("\nCleaning complete.")
print(f"Total products after cleaning and filtering: {len(df_final)}")
print("\nValue counts for new 'type' column:")
print(df_final['type'].value_counts())