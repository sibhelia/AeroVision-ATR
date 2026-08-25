import os
import glob
import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from ultralytics import YOLO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Aşama: YOLOv8 Nesne Tespiti Modeli
yolo_paths = [
    "models/yolo_runs/flir_yolov8n/weights/best.pt",
    "runs/detect/flir_yolov8n/weights/best.pt",
    "runs/detect/val/weights/best.pt",
    "runs/detect/train/weights/best.pt"
]
yolo_path = next((p for p in yolo_paths if os.path.exists(p)), None)

if yolo_path:
    detector = YOLO(yolo_path)
    print(f"[OK] YOLOv8 tespit modeli yüklendi: {yolo_path}")
else:
    raise FileNotFoundError("YOLO best.pt ağırlık dosyası bulunamadı.")

# 2. Aşama: MobileNetV3 Hedef Doğrulama Modeli (.pth)
cls_paths = [
    "best_mobilenetv3.pth",
    "models/best_mobilenetv3.pth",
    "models/checkpoints/best_mobilenetv3.pth"
]
classifier_path = next((p for p in cls_paths if os.path.exists(p)), None)
classifier = None

if classifier_path:
    try:
        classifier = models.mobilenet_v3_large(weights=None)
        classifier.classifier[0] = nn.Linear(960, 1024)
        classifier.classifier[3] = nn.Linear(1024, 2)
        
        checkpoint = torch.load(classifier_path, map_location=device, weights_only=True)
        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        
        classifier.load_state_dict(state_dict)
        classifier.to(device)
        classifier.eval()
        print(f"[OK] MobileNetV3 doğrulama modeli yüklendi: {classifier_path}")
    except Exception as e:
        print(f"[HATA] MobileNetV3 yüklenirken hata oluştu: {e}")
        classifier = None
else:
    print("[UYARI] best_mobilenetv3.pth bulunamadı, yalnızca YOLO çıkarımı yapılacak.")

# Sınıflandırıcı için görüntü önişleme
cls_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Sınıf indeks haritaları
YOLO_CLASSES = {0: 'car', 1: 'person'}
CLS_CLASSES = {0: 'person', 1: 'car'}

def run_two_stage_atr(image_path, conf_thresh=0.25):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Görsel okunamadı: {image_path}")
        return

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Adım 1: YOLO ile Tespit
    yolo_results = detector.predict(source=image_rgb, conf=conf_thresh, verbose=False)[0]
    boxes = yolo_results.boxes
    
    print(f"\n--- Analiz: {os.path.basename(image_path)} | Tespit Edilen Hedef: {len(boxes)} ---")
    
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        det_cls = int(box.cls[0].item())
        det_conf = float(box.conf[0].item())
        
        h, w, _ = image_rgb.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        crop = image_rgb[y1:y2, x1:x2]
        det_label = YOLO_CLASSES.get(det_cls, 'target')
        final_label = det_label
        final_conf = det_conf
        
        # Adım 2: MobileNetV3 ile Doğrulama (Verification)
        if classifier is not None and crop.size > 0:
            pil_crop = Image.fromarray(crop)
            tensor_crop = cls_transforms(pil_crop).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = classifier(tensor_crop)
                probs = torch.softmax(logits, dim=1)
                cls_idx = torch.argmax(probs, dim=1).item()
                cls_conf = probs[0][cls_idx].item()
                
            verified_name = CLS_CLASSES.get(cls_idx, 'target')
            final_label = f"{verified_name} (Dogrulandi)"
            final_conf = cls_conf

        print(f"Hedef {idx+1}: [{det_label}] -> {final_label} | Güven: %{final_conf * 100:.1f} | Kutu: [{x1}, {y1}, {x2}, {y2}]")
        
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_bgr, f"{final_label} %{final_conf*100:.0f}", (x1, max(y1 - 10, 15)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", f"atr_out_{os.path.basename(image_path)}")
    cv2.imwrite(out_path, image_bgr)
    print(f"Sonuç kaydedildi: {out_path}")

if __name__ == "__main__":
    test_imgs = glob.glob("dataset/yolo/images/val/*.jpg")
    if test_imgs:
        run_two_stage_atr(test_imgs[0])
    else:
        print("Test için görsel bulunamadı.")