import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
import numpy as np
from skimage.feature import local_binary_pattern
from PIL import Image
from collections import Counter

# ============================
# 1. Device
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================
# 2. Paths
# ============================
data_dir = r"C:\Users\aksha\Downloads\project_dataset\dataset\dataset_split"

# ============================
# 3. Custom LBP Transform
# ============================
class LBPTransform:
    def __init__(self, P=8, R=1):
        self.P = P
        self.R = R

    def __call__(self, img):
        img_gray = img.convert("L")
        np_img = np.array(img_gray)
        lbp = local_binary_pattern(np_img, self.P, self.R, method="uniform")
        lbp = (lbp / (lbp.max() + 1e-8) * 255).astype(np.uint8)
        return Image.fromarray(lbp)

# ============================
# 4. Transforms
# ============================
rgb_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

lbp_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    LBPTransform(P=8, R=1),
    transforms.ToTensor(),   # 1-channel tensor
    transforms.Normalize([0.5], [0.5])
])

# ============================
# 5. Custom Dataset (RGB + LBP)
# ============================
class DualDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)

        rgb_img = rgb_transform(img)
        lbp_img = lbp_transform(img)

        return (rgb_img, lbp_img), target

# ============================
# 6. Model Definition with SE & Diff Fusion
# ============================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x)
        return x * w

class DualEfficientNetSE(nn.Module):
    def __init__(self, num_classes):
        super(DualEfficientNetSE, self).__init__()
        self.rgb_model = EfficientNet.from_pretrained("efficientnet-b0")
        self.rgb_features = self.rgb_model._fc.in_features
        self.rgb_model._fc = nn.Identity()

        self.lbp_model = EfficientNet.from_pretrained("efficientnet-b0")
        self.lbp_features = self.lbp_model._fc.in_features
        self.lbp_model._fc = nn.Identity()
        self.lbp_model._conv_stem = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

        fused_dim = self.rgb_features + self.lbp_features + self.rgb_features  # rgb + lbp + abs diff

        self.proj = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.se = SEBlock(1024, reduction=16)

        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, rgb, lbp):
        rgb_feat = self.rgb_model(rgb)
        lbp_feat = self.lbp_model(lbp)
        diff = torch.abs(rgb_feat - lbp_feat)
        fused = torch.cat([rgb_feat, lbp_feat, diff], dim=1)
        x = self.proj(fused)
        x = self.se(x)
        out = self.classifier(x)
        return out

# ============================
# Main execution: training + testing
# ============================
if __name__ == "__main__":
    # Datasets & Loaders
    train_dataset = DualDataset(os.path.join(data_dir, "train"))
    val_dataset   = DualDataset(os.path.join(data_dir, "val"))
    test_dataset  = DualDataset(os.path.join(data_dir, "test"))

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print("Classes:", class_names)

    # Sampler for class imbalance
    targets = train_dataset.targets
    class_counts = Counter(targets)
    sample_weights = [1.0 / class_counts[t] for t in targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model, Loss, Optimizer, Scheduler
    model = DualEfficientNetSE(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    # Training Loop
    num_epochs = 12
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for (rgb, lbp), labels in train_loader:
            rgb, lbp, labels = rgb.to(device), lbp.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(rgb, lbp)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for (rgb, lbp), labels in val_loader:
                rgb, lbp, labels = rgb.to(device), lbp.to(device), labels.to(device)
                outputs = model(rgb, lbp)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_dataset)
        val_acc = val_corrects.double() / len(val_dataset)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_dual_effnet_se.pth")
            print("✅ Saved best model")

    print("Training complete!")

    # Testing & reporting
    model.load_state_dict(torch.load("best_dual_effnet_se.pth", map_location=device))
    model.eval()
    test_corrects = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for (rgb, lbp), labels in test_loader:
            rgb, lbp, labels = rgb.to(device), lbp.to(device), labels.to(device)
            outputs = model(rgb, lbp)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = test_corrects.double() / len(test_dataset)
    print("Test Accuracy: {:.4f}".format(test_acc))

    from sklearn.metrics import confusion_matrix, classification_report
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(all_labels, all_preds)
    print("Classification Report:\n", classification_report(all_labels, all_preds, target_names=class_names))

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
