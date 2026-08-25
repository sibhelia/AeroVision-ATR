# AeroVision-ATR: Real-Time Airborne Thermal Target Recognition & Benchmark Pipeline

AeroVision-ATR is an end-to-end edge-AI computer vision framework engineered for thermal infrared (LWIR/FLIR) airborne surveillance and autonomous unmanned systems. The project integrates an empirical architectural benchmark across lightweight vision backbones with an optimized real-time target detection and verification pipeline.

---

## Key Architectural Components

1. **Empirical Edge-AI Benchmark Engine (`src/engine.py`, `src/benchmark.py`)**
   - Comparative performance evaluation across **ResNet-18**, **MobileNetV3-Large**, and **DenseNet-121** under FLIR thermal domain constraints.
   - Comprehensive profiling of inference latency (ms), frame rates (FPS), FLOPs, parameter footprint, and validation accuracy.
   - Class-weighted Cross-Entropy loss integration to resolve vehicle/person thermal target imbalances.

2. **Real-Time Detection & Spatial Localization (`src/train_yolo.py`)**
   - Single-stage lightweight **YOLOv8 Nano** detector fine-tuned on full-frame FLIR infrared imagery.
   - High-throughput spatial bounding-box regression achieving **%79.86 mAP@50** at sub-4ms inference latency (~275+ FPS on edge-class GPUs).

3. **Two-Stage ATR Inference Pipeline (`src/inference.py`)**
   - Full-frame real-time candidate target proposal via YOLOv8.
   - Dynamic region-of-interest (RoI) extraction and secondary class verification using fine-tuned backbones to suppress false alarms in low-contrast night vision scenes.

---

## Benchmark & Detection Performance

| Model Architecture | Task | Input Resolution | Val Accuracy / mAP@50 | Latency (Inference) | Edge Suitability |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MobileNetV3-Large** | Target Classification | 224x224 | Benchmark Baseline | Low Memory Footprint | High (Primary) |
| **ResNet-18** | Target Classification | 224x224 | Benchmark Baseline | Balanced Latency | Medium |
| **DenseNet-121** | Target Classification | 224x224 | Benchmark Baseline | Feature Dense | Low |
| **YOLOv8n (FLIR-ATR)** | Target Detection | 640x640 | **%79.86 mAP@50** | **~3.6 ms** | **Ultra-High (Real-Time)** |

---

## Project Structure

```text
AeroVision-ATR/
├── dataset/
│   ├── yolo/                 # YOLO formatted train/val images & labels
│   └── crops/                # Cropped target benchmarks
├── models/
│   ├── checkpoints/          # PyTorch classification weights
│   └── yolo_runs/            # YOLOv8 training runs and metrics
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory thermal data analysis
│   ├── 06_benchmark.ipynb    # Latency & throughput profiling
│   └── 08_evaluate_yolo.ipynb# YOLO metric validation & test visuals
├── outputs/                  # Real-time inference annotated frames
├── src/
│   ├── dataset.py            # Custom PyTorch thermal dataset loaders
│   ├── models.py             # Backbone model definitions
│   ├── engine.py             # Modular training & validation loops
│   ├── train_yolo.py         # YOLO headless training script
│   └── inference.py          # Two-stage ATR end-to-end pipeline
├── BENCHMARK_REPORT.md       # Full hardware benchmark report
└── README.md
