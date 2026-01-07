import cv2
import numpy as np
import onnxruntime as ort
import os
from pathlib import Path
from tqdm import tqdm

CONFIG = {
    "model_path": "checkpoints/best_unet_model.onnx",
    "dataset_root": "/home/inonzr/datasets/mvtec_anomaly_detection/cable",
    "image_size": 256,
    "threshold": 0.5,
    "min_defect_area": 50
}

def calculate_dice(pred, target):
    pred = (pred > CONFIG["threshold"]).astype(np.float32)
    target = (target > 0.5).astype(np.float32)
    intersection = (pred * target).sum()
    dice = (2. * intersection) / (pred.sum() + target.sum() + 1e-7)
    return dice


def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img, (CONFIG["image_size"], CONFIG["image_size"]))

    # נרמול (זהה למה שעשית באימון)
    img_input = resized.astype(np.float32) / 255.0
    img_input = (img_input - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img_input = img_input.transpose(2, 0, 1)
    img_input = np.expand_dims(img_input, axis=0).astype(np.float32)
    return img_input, resized


def run_evaluation():
    session = ort.InferenceSession(CONFIG["model_path"],
                                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    test_dir = Path(CONFIG["dataset_root"]) / "test"
    gt_dir = Path(CONFIG["dataset_root"]) / "ground_truth"

    results = []

    print(f"🔍 Starting evaluation on: {CONFIG['dataset_root']}")

    for defect_type in test_dir.iterdir():
        if not defect_type.is_dir(): continue

        images = list(defect_type.glob("*.png"))
        for img_path in tqdm(images, desc=f"Testing {defect_type.name}"):
            img_input, _ = preprocess_image(str(img_path))

            outputs = session.run(None, {input_name: img_input})
            pred_mask = 1 / (1 + np.exp(-outputs[0][0][0]))  # Sigmoid

            dice_score = 0
            is_actually_defective = (defect_type.name != "good")

            if is_actually_defective:
                mask_path = gt_dir / defect_type.name / (img_path.stem + "_mask.png")
                if mask_path.exists():
                    gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    gt_mask = cv2.resize(gt_mask, (CONFIG["image_size"], CONFIG["image_size"])) / 255.0
                    dice_score = calculate_dice(pred_mask, gt_mask)

            predicted_defective = np.sum(pred_mask > CONFIG["threshold"]) > CONFIG["min_defect_area"]

            results.append({
                "category": defect_type.name,
                "dice": dice_score,
                "correct_decision": (predicted_defective == is_actually_defective)
            })

    total_imgs = len(results)
    avg_dice = np.mean([r['dice'] for r in results if r['category'] != 'good'])
    accuracy = np.mean([r['correct_decision'] for r in results]) * 100

    print("\n" + "=" * 30)
    print(" MODEL RELIABILITY REPORT")
    print("=" * 30)
    print(f"Total Images Tested:  {total_imgs}")
    print(f"Classification Accuracy: {accuracy:.2f}%")
    print(f"Mean Dice Score (Defects): {avg_dice:.4f}")
    print("=" * 30)


if __name__ == "__main__":
    run_evaluation()