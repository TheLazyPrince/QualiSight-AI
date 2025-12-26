import cv2
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
import os


def calculate_dice(pred, target, threshold=0.5):
    pred = (pred > threshold).astype(np.float32)
    target = (target > 0.5).astype(np.float32)
    intersection = (pred * target).sum()
    dice = (2. * intersection) / (pred.sum() + target.sum() + 1e-7)
    return dice


def evaluate_defect(model_path, image_path, mask_path=None):
    session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    orig_img = cv2.imread(image_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(orig_img, (256, 256))

    img_input = resized.astype(np.float32) / 255.0
    img_input = (img_input - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img_input = img_input.transpose(2, 0, 1)
    img_input = np.expand_dims(img_input, axis=0).astype(np.float32)

    outputs = session.run(None, {input_name: img_input})
    pred_mask = 1 / (1 + np.exp(-outputs[0][0][0]))

    dice_score = None
    if mask_path and os.path.exists(mask_path):
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.resize(gt_mask, (256, 256)) / 255.0
        dice_score = calculate_dice(pred_mask, gt_mask)

    is_defective = np.sum(pred_mask > 0.5) > 50
    result_text = "FAIL (Defect Detected)" if is_defective else "PASS (Good)"

    heatmap = cv2.applyColorMap(np.uint8(255 * pred_mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(resized, 0.6, heatmap, 0.4, 0)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(resized);
    plt.title("Original Image")
    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask, cmap='hot');
    plt.title(f"Predicted Mask\nDice: {f'{dice_score:.4f}' if dice_score else 'N/A'}")
    plt.subplot(1, 3, 3)
    plt.imshow(overlay);
    plt.title(f"Decision: {result_text}")
    plt.show()


if __name__ == "__main__":
    evaluate_defect(
        "checkpoints/best_unet_model.onnx",
        "D:\\dataset\\mvtec_anomaly_detection\\cable\\test\\bent_wire\\000.png",
        "D:\\dataset\\mvtec_anomaly_detection\\cable\\ground_truth\\bent_wire\\000_mask.png"
    )