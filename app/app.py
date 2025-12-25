import gradio as gr
import pandas as pd
import numpy as np
import os
import ast
from PIL import Image

from sentence_transformers import SentenceTransformer, util
import spacy
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
YOLO_PATH = os.path.join(ROOT_DIR, "output", "best.pt")
RECIPE_DATA_PATH = os.path.join(ROOT_DIR, "output", "recipes_final_with_images.parquet")
EMBEDDINGS_PATH = os.path.join(ROOT_DIR, "output", "recipe_embeddings.npy")
IMAGES_DIR = os.path.join(ROOT_DIR, "output", "Images") 

def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def load_data():
    if not os.path.exists(RECIPE_DATA_PATH):
        raise FileNotFoundError(f"Data not found at: {RECIPE_DATA_PATH}")
    
    df = pd.read_parquet(RECIPE_DATA_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)
    return df, embeddings

model = load_model()
df, recipe_embeddings = load_data()

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")
    
def normalize_ingredients(ingredients):
    doc = nlp(" ".join(ingredients))
    return [
        t.lemma_.lower().strip()
        for t in doc
        if t.pos_ == "NOUN" and not t.is_stop
    ]

def recommend_recipes(user_ingredients, top_n=10):
    tokens = normalize_ingredients(user_ingredients)
    if not tokens:
        return pd.DataFrame()

    user_embedding = model.encode(" ".join(tokens), convert_to_tensor=True)
    cosine_scores = util.cos_sim(user_embedding, recipe_embeddings)[0].cpu().numpy()

    rows = []
    for i, score in enumerate(cosine_scores):
        recipe_tokens_raw = df.iloc[i]["Ingredient_Final"]
        if isinstance(recipe_tokens_raw, str):
             try:
                 recipe_tokens = set(ast.literal_eval(recipe_tokens_raw))
             except (ValueError, SyntaxError):
                 recipe_tokens = set()
        else:
             recipe_tokens = set(recipe_tokens_raw)
             
        overlap = len(set(tokens) & recipe_tokens) / max(len(tokens), 1)
        final_score = 0.5 * score + 0.5 * overlap

        rows.append({
            "Title": df.iloc[i]["Title"],
            "Image_Name": df.iloc[i]["Image_Path"], 
            "Ingredients": df.iloc[i]["Ingredients"],
            "Instructions": df.iloc[i]["Instructions"],
            "Similarity": float(score),
            "Overlap": float(overlap),
            "FinalScore": float(final_score),
        })

    out = pd.DataFrame(rows)
    return out.sort_values("FinalScore", ascending=False).head(top_n).reset_index(drop=True)

def parse_ingredients_cell(val):
    if isinstance(val, list):
        items = val
    elif isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                items = parsed
            else:
                items = [parsed]
        except:
            lines = [l.strip() for l in val.splitlines() if l.strip()]
            if len(lines) > 1:
                items = lines
            else:
                items = [x.strip() for x in val.split(",") if x.strip()]
    else:
        items = []

    cleaned = []
    for x in items:
        if not isinstance(x, str):
            x = str(x)
        x = " ".join(x.replace("\n", " ").split()).strip()
        if x:
            cleaned.append(x)
    return cleaned

def get_recipe_image_path(image_path_stub):
    if not isinstance(image_path_stub, str):
        return None

    for ext in [".jpg", ".jpeg", ".png"]:
        full_path = os.path.join(ROOT_DIR, "output", image_path_stub + ext)
        if os.path.exists(full_path):
            return full_path
        alt_path = os.path.join(IMAGES_DIR, os.path.basename(image_path_stub) + ext)
        if os.path.exists(alt_path):
            return alt_path
    
    return None

def load_cv_model():
    if not os.path.exists(YOLO_PATH):
        local_path = os.path.join(BASE_DIR, "best.pt")
        if os.path.exists(local_path):
            return YOLO(local_path)
        raise FileNotFoundError(f"YOLO model not found at: {YOLO_PATH}")
    return YOLO(YOLO_PATH)

cv_model = load_cv_model()

def detect_ingredients(pil_image, conf=0.3, iou=0.6):
    result = cv_model.predict(pil_image, conf=conf, iou=iou, verbose=False)[0]
    detected = sorted({ cv_model.names[int(box.cls[0])] for box in result.boxes })
    plotted = result.plot()
    return detected, Image.fromarray(plotted[..., ::-1])

def gr_detect_and_update(image, manual_text):
    if image is None:
        return [], None, manual_text

    pil = Image.fromarray(image).convert("RGB")
    detected_list, plotted = detect_ingredients(pil)

    manual_list = [x.strip() for x in manual_text.split(",") if x.strip()]
    
    seen = set()
    combined_items = []
    
    for item in manual_list + detected_list:
        t = item.lower().strip()
        if t and t not in seen:
            seen.add(t)
            combined_items.append(t)
            
    combined_str = ", ".join(combined_items)
    
    return detected_list, plotted, combined_str


def gr_recommend(manual_text, detected_list):
    manual_list = [x.strip() for x in manual_text.split(",") if x.strip()]

    combined = []
    seen = set()

    for item in manual_list + (detected_list or []):
        t = item.lower().strip()
        if t and t not in seen:
            seen.add(t)
            combined.append(t)
            
    if not combined:
        return "<h3 style='color:red;'>Error: Please enter ingredients manually or run detection first.</h3>"

    results = recommend_recipes(combined)

    if results.empty:
        return "No recipes found matching your input."

    output = ""
    for i, row in results.iterrows():
        image_path = get_recipe_image_path(row['Image_Name']) 
        
        output += f"### {i+1}. {row['Title']}\n"
        
        #HTML Tag (Using the relative path and file= protocol)
        # if image_path:
        #     output += f'<img src="file={image_path}" alt="{row["Title"]}" width="300px"/>\n\n'
        # else:
        #     output += f"*(No image available)*\n\n"
        
        output += f"**Similarity:** {row['Similarity']:.4f}\n\n"
        output += f"**Overlap:** {row['Overlap']:.4f}\n\n"
        output += f"**Final Score:** {row['FinalScore']:.4f}\n\n"
        output += f"**Ingredients:**\n"
        for ing in parse_ingredients_cell(row["Ingredients"]):
            output += f"- {ing}\n"
        output += f"\n**Instructions:**\n{row['Instructions']}\n\n---\n"

    return output


with gr.Blocks(title="Recipe Recommendation System") as demo:
    gr.Markdown("#Recipe Recommendation System")
    
    detected_state = gr.State(value=[])
    
    with gr.Row():
        manual_text = gr.Textbox(
            label="1. Enter Ingredients (comma separated)",
            placeholder="e.g., onion, potato, chicken, rice",
            lines=3
        )
        
    with gr.Row():
        img_input = gr.Image(label="2. Upload Image for Detection", type="numpy")
        plotted_output = gr.Image(label="Detection Result")
        
    with gr.Row():
        detect_btn = gr.Button("Run Detection")
        recommend_btn = gr.Button("Find Recipes")

    recipe_output = gr.Markdown()

    detect_btn.click(
        fn=gr_detect_and_update,
        inputs=[img_input, manual_text],
        outputs=[detected_state, plotted_output, manual_text]
    )

    recommend_btn.click(
        fn=gr_recommend,
        inputs=[manual_text, detected_state],
        outputs=[recipe_output]
    )

if __name__ == "__main__":
    demo.launch()