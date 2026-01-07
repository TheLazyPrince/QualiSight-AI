import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

CONFIG = {
    "image_size": 256,
    "batch_size": 16,
    "epochs": 100,
    "lr": 1e-4,
    "base_path": "/home/inonzr/datasets/mvtec_anomaly_detection",
    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
}

train_transform = A.Compose([
    A.Resize(CONFIG["image_size"], CONFIG["image_size"]),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results_visuals", exist_ok=True)


class MVTecSegmentation(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        self.mask_paths = []

        for category in [d.name for d in self.root_dir.iterdir() if d.is_dir()]:
            test_dir = self.root_dir / category / "test"
            gt_dir = self.root_dir / category / "ground_truth"
            if not test_dir.exists(): continue

            for defect_type in test_dir.iterdir():
                if defect_type.is_dir():
                    for img_path in defect_type.glob("*.png"):
                        self.image_paths.append(str(img_path))
                        if defect_type.name == "good":
                            self.mask_paths.append(None)
                        else:
                            mask_path = gt_dir / defect_type.name / (img_path.stem + "_mask.png")
                            self.mask_paths.append(str(mask_path) if mask_path.exists() else None)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.mask_paths[idx] is None:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        else:
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            mask = (mask > 0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        return image, mask.unsqueeze(0)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x): return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, 2, 2))
            self.ups.append(DoubleConv(feature * 2, feature))
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skips = skip_connections[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i // 2]
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            x = self.ups[i + 1](torch.cat((skip, x), dim=1))
        return self.final_conv(x)


class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, smooth=1):
        bce_loss = self.bce(inputs, targets)
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return bce_loss + dice_loss


def save_visual_results(model, loader, device, epoch):
    model.eval()
    images, targets = next(iter(loader))
    images = images.to(device)
    with torch.no_grad():
        preds_probs = torch.sigmoid(model(images))

    img_np = images[0].cpu().permute(1, 2, 0).numpy()
    img_np = np.clip(img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
    pred_mask = preds_probs[0].cpu().squeeze().numpy()

    heatmap = cv2.applyColorMap(np.uint8(255 * pred_mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(np.uint8(255 * img_np), 0.6, heatmap, 0.4, 0)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img_np);
    ax[0].set_title("Original")
    ax[1].imshow(targets[0].cpu().squeeze().numpy(), cmap='gray');
    ax[1].set_title("Ground Truth")
    ax[2].imshow(overlay);
    ax[2].set_title(f"Heatmap Ep {epoch + 1}")
    for a in ax: a.axis('off')
    plt.savefig(f"results_visuals/epoch_{epoch + 1}.png")
    plt.close()


def export_onnx(model, file_path, device):
    model.eval()
    dummy_input = torch.randn(1, 3, CONFIG["image_size"], CONFIG["image_size"]).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        file_path,
        export_params=True,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f" Model exported successfully to ONNX: {file_path}")


def main():
    dataset = MVTecSegmentation(CONFIG["base_path"], transform=train_transform)
    train_loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True,
                              num_workers=8, pin_memory=True, persistent_workers=True)

    model = UNet().to(CONFIG["device"])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    criterion = DiceBCELoss()
    scaler = torch.amp.GradScaler('cuda')

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    best_loss = float('inf')

    print(f"Starting 50 Epoch Training on {CONFIG['device']}...")
    for epoch in range(CONFIG["epochs"]):
        model.train()
        epoch_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{CONFIG['epochs']}]")

        for data, targets in loop:
            data, targets = data.to(CONFIG["device"]), targets.to(CONFIG["device"])

            with torch.amp.autocast('cuda'):
                preds = model(data)
                loss = criterion(preds, targets)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/best_unet_model.pth")
            print(f"--> New Best Model Saved! Loss: {best_loss:.4f}")

        if (epoch + 1) % 5 == 0 or epoch == 0:
            save_visual_results(model, train_loader, CONFIG["device"], epoch)

    print("\nTraining finished successfully.")

    print("Preparing to export the best model to ONNX...")
    torch.cuda.empty_cache()

    best_model = UNet().to(CONFIG["device"])
    checkpoint_path = "checkpoints/best_unet_model.pth"

    if os.path.exists(checkpoint_path):
        best_model.load_state_dict(torch.load(checkpoint_path, map_location=CONFIG["device"]))
        onnx_path = "checkpoints/best_unet_model.onnx"
        export_onnx(best_model, onnx_path, CONFIG["device"])
    else:
        print("Error: Best weights file not found!")


if __name__ == "__main__":
    main()