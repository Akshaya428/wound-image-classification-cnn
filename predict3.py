import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import local_binary_pattern

# ---- Import model class from your training file ----
from train3 import DualEfficientNetSE  # Ensure train3.py is in the same folder

# ---- Device ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---- Load the trained model ----
model = DualEfficientNetSE(num_classes=3)
model.load_state_dict(torch.load("best_dual_effnet_se.pth", map_location=device))
model.to(device)
model.eval()

# ---- Class names (ensure order matches your training dataset) ----
class_names = ['DFU', 'Normal', 'PressureUlcer']
display_names = {
    'DFU': 'Diabetic Foot Ulcer',
    'Normal': 'Normal',
    'PressureUlcer': 'Other Wound'
}

# ---- Custom LBP Transform ----
def compute_lbp_image(pil_img, P=8, R=1):
    gray = pil_img.convert("L")
    np_img = np.array(gray)
    lbp = local_binary_pattern(np_img, P, R, method="uniform")
    lbp = (lbp / lbp.max() * 255).astype(np.uint8)
    return Image.fromarray(lbp)

# ---- Transforms ----
rgb_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

lbp_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ---- Prediction Function ----
def predict_image(image_path):
    # Load the original image
    img = Image.open(image_path).convert("RGB")

    # Compute the LBP version
    lbp_img = compute_lbp_image(img)

    # Transform both
    rgb_tensor = rgb_transform(img).unsqueeze(0).to(device)
    lbp_tensor = lbp_transform(lbp_img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(rgb_tensor, lbp_tensor)
        probs = F.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    # Map prediction
    pred_label = class_names[pred.item()]
    pred_display = display_names[pred_label]
    confidence = conf.item() * 100

    # ---- Display the image with label & confidence ----
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"{pred_display}\nConfidence: {confidence:.2f}%", fontsize=14, fontweight="bold")
    plt.show()

    print(f"Predicted Class: {pred_display}")
    print(f"Confidence: {confidence:.2f}%")

# ---- Example usage ----
predict_image(r"C:\Users\aksha\Downloads\project_dataset\sample2.png")  # Change to your image path
