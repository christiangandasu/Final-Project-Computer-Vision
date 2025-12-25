# Deep Learning Based Recipe Suggestion with Image Recognition

A practical application using Sentence-Transformers for semantic search and Ultralytics YOLO for ingredient detection, allowing users to find recipes they can cook immediately based on available ingredients.


## 1. Live Application

The application is deployed and accessible here:

**https://huggingface.co/spaces/gandasu/Computer_Vision_Project**

## 2. Key Features

The system provides recipe ranking based on a Balanced-Focus approach:

* Intelligent Search: Combines manual text input and image-detected ingredients.
* Ranking Formula: Prioritizes both conceptual relevance.

$$\text{Final Score} = 0.5 \times \text{Semantic Similarity} + 0.5 \times \text{Ingredient Overlap}$$

| Component | Technology | Core Function |
| :--- | :--- | :--- |
| **Semantic Search** | `all-MiniLM-L6-v2` (Sentence-Transformers) | Generates 384-dim embeddings to measure conceptual relevance. |
| **Object Detection** | `ultralytics` (YOLO) | Real-time ingredient recognition using the custom model (`best.pt`). |
| **Preprocessing** | `spaCy` (`en-core-web-sm`) | Standardizes input ingredients via lemmatization. |

## 3. Repository Structure

| File / Folder | Purpose |
| :--- | :--- |
| **`app.py`** | Main application file. |
| **`requirements.txt`** | Complete dependency list. |
| **`best.pt`** | YOLO model file. |
| **`recipes_final_with_images.parquet`** | Main recipe metadata dataset. |
| **`recipe_embeddings.npy`** | Pre-calculated embeddings for the recipe database. |
| **`NLP_Model.ipynb` / `CV_Model.ipynb`** | Development and exploratory notebooks. |


## 4. Local Setup and Execution

To run this application locally, follow these steps:

1.  **Clone the Repository:**
    ```bash
    # Step 1a: Clone the repository to your local machine
    git clone [https://github.com/christiangandasu/Final-Project-Deep-Learning](https://github.com/christiangandasu/Final-Project-Deep-Learning)

    # Step 1b: Navigate into the newly created project folder
    cd Final-Project-Computer-Vision
    ```
    
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    
3.  **Run the Application:**
    ```bash
    python app.py
    ```
    The application will launch in your browser at the local URL provided in the console (e.g., `http://127.0.0.1:7860`).

---

**Authors:** [Alexander Christian Gandasurya, Samuel Jeremy Winoto, Lauw Samuel Lelono, Yonathan Immanuel Nurhan]
