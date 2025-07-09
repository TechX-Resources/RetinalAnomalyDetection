import os
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles

from pathlib import Path
from inference import run_inference
from checker import is_retinal_image_openai
from explainer import explain_prediction
from utils import load_data, category_percentage, correlation_between_labels, venn_diagram

"""
< Unifying (Abstration) for easy implementation and retrieve using simple calls as follows: >
<   - model.explain()           - model.tuned_predict()              - model.random_pick()#for random test image pick       
"""

class Model:
    def __init__(self, image=""):
        if image == "":
            image = self.random_pick()
        self.path = Path(image) if image else None

    def predict(self):
        if not self.path:
            raise ValueError("No image path set.")
        predictions, _ = run_inference(str(self.path))
        return predictions
    
    def tuned_predict(self):
        for label, prob in self.predict():
            print(f"{label}: {100 * prob:.3f}%")
        print()

    def explain(self):
        return str(explain_prediction(self.predict()))

    def random_pick(self, data="data/test"):
        """
        If no image, then pick a random image from the test set
        """
        try:
            all_entries = os.listdir(data)
            files = [entry for entry in all_entries if os.path.isfile(os.path.join(data, entry))]
            if not files:
                print(f"No files found in '{data}'.")
                return None
            random_filename = random.choice(files)
            self.path = os.path.join(data, random_filename)
            return self.path
        except FileNotFoundError:
            print(f"Error: Directory not found at '{data}'.")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
    
    def data_visual(self) -> int:
        try: 
            from utils import load_data, category_percentage, correlation_between_labels, venn_diagram
        except ImportError as ie:
            print("Import Error :", str(ie))
            return -1

        
        df = load_data("data/train/train.csv")

        if isinstance(df, pd.DataFrame):
            categories = [
                'diabetic retinopathy',
                'glaucoma',
                'macular edema',
                'macular degeneration',
                'retinal vascular occlusion',
                'normal'
            ]

            category_percentage(df, categories)
            correlation_between_labels(df)

            G1 = [0, 1, 2]  # retinopathy, glaucoma, edema
            G2 = [1, 3, 4]  # glaucoma, degeneration, occlusion
            G3 = [0, 4, 5]  # retinopathy, occlusion, normal
            G4 = [2, 3, 5]  # edema, degeneration, normal
            venn_diagram(df, categories, G1, G2, G3, G4)
            return 

        else:
            print("Failed to load DataFrame.")
        



#image_path = "data/test/0a2229abced7.jpg" for testing
model = Model()
model.random_pick()

if is_retinal_image_openai(model.path): 
    print("Prediction ...")
    print(model.tuned_predict())
    print()
    print("English ...")
    print(model.explain())
    print()
    print("Data Visualization ...")
    print(model.data_visual() if (model.data_visual()) != -1 else "")

else:
    print("Not a valid retinal image.")

