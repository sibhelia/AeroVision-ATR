import os
import yaml
import torch
from ultralytics import YOLO

def create_yaml(base_dir):
    yaml_path = os.path.join(base_dir, 'data.yaml')
    yaml_content = {
        'path': base_dir,
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'car',
            1: 'person'
        }
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    return yaml_path

def train():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    yolo_dir = os.path.join(project_root, 'dataset', 'yolo')
    yaml_path = create_yaml(yolo_dir)
    
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Cihaz: {device} | YAML: {yaml_path}")
    
    model = YOLO('yolov8n.pt')
    
    model.train(
        data=yaml_path,
        epochs=25,
        imgsz=640,
        batch=16,
        device=device,
        project=os.path.join(project_root, 'models', 'yolo_runs'),
        name='flir_yolov8n',
        save=True,
        workers=2,
        plots=True
    )

if __name__ == '__main__':
    train()