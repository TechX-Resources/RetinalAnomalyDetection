
# Retinal Disease Classification – Data Analysis

This document contains extended analysis and visualizations that support the Updates.md in `Updates.md`.

[⬅ Back to Updates](./Updates.md)

---

## Sample Data 

Showing the data shape and format in order to be able to internally visualize and structure it.
<img width="449" height="574" alt="Capture d’écran 2025-07-22 à 17 04 43" src="https://github.com/user-attachments/assets/b181e25c-52e1-4708-a6fc-cf67b72e9349" />

---

## GLCM Texture Feature Analysis

Exploring how Gray-Level Co-occurrence Matrix (GLCM) features influence classification performance.

<img width="998" height="491" alt="Capture d’écran 2025-07-20 à 22 34 52" src="https://github.com/user-attachments/assets/4709f3c6-e29b-4204-a355-42335b0f0bbd" />
<img width="995" height="483" alt="Capture d’écran 2025-07-20 à 22 35 06" src="https://github.com/user-attachments/assets/61f5b9f5-a454-4a7e-b400-21c571ca0429" />
<img width="988" height="489" alt="Capture d’écran 2025-07-20 à 22 34 02" src="https://github.com/user-attachments/assets/443ab43c-fa84-4172-9832-f3bd6a5022b8" />

- Contrast, Homogeneity metrics, Dissimilarity

---

## RBG  & Spectrum Feature Analysis (Planned)

- Brightness Distribution
<img width="461" height="214" alt="Capture d’écran 2025-07-22 à 17 07 14" src="https://github.com/user-attachments/assets/e532b699-2c61-4427-a0b5-b8e2cff2cdc1" />

- Color spectrum feature contribution and class separability (RBG)
<img width="728" height="316" alt="Capture d’écran 2025-07-22 à 17 08 34" src="https://github.com/user-attachments/assets/b6ad0a88-c490-48e7-9195-fce151067717" />

- Sharpness Distribution
<img width="716" height="366" alt="Capture d’écran 2025-07-22 à 17 09 49" src="https://github.com/user-attachments/assets/e4199d20-4226-482c-bfb5-ad93f14eb47f" />


---

## Augmentation Impact

Evaluating how various augmentations affect model robustness:

- Flip, Rotate, ColorJitter, Crop
- Discussion of augmentation-heavy vs. clean dataset performance

```python
img_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}
```

---

