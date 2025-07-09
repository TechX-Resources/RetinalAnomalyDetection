import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import warnings


"""
< Image Preprocessing and Transformation (Application) >
< 
    The preprocessing steps applied to input images before 
    feeding them into the model. It converts images to tensors and normalizes 
    them using the ImageNet mean and standard deviation, which the ResNet model 
    expects for accurate inference.
>
"""

# Ignore torchvision deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)

MODEL_PATH = 'models/model.pth'
LABELS_CSV = 'data/train/train.csv'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(LABELS_CSV)
disease_labels = df.columns[1:].tolist()

img_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_model(num_labels):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_labels)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def predict_image(model, image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = img_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output).cpu().numpy()[0]

    results = list(zip(disease_labels, probs))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def explain_prediction(results):
    top_label, top_prob = results[0]
    explanation = f"The image is most likely to show: {top_label} with confidence {top_prob:.2f}."
    if top_prob < 0.5:
        explanation += " However, confidence is low, so it may be normal or uncertain."
    return explanation

def get_predictions(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' does not exist.")
    model = load_model(len(disease_labels))
    predictions = predict_image(model, image_path)
    return predictions

def get_explanation(predictions):
    return explain_prediction(predictions)

def run_inference(image_path):
    try:
        predictions = get_predictions(image_path)
    except FileNotFoundError as e:
        return str(e)

    explanation = get_explanation(predictions)

    return predictions, explanation


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python inference.py <path_to_image>")
        sys.exit(1)

    preds, explanation = run_inference(sys.argv[1])
    print("Predictions (disease: confidence):")
    for label, prob in preds:
        print(f"{label}: {prob:.3f}")
    print()
    print(explanation)
