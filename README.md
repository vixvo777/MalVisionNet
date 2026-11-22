# MalVisionNet - Windows PE Malware Detection using Binary-to-Image Learning

**Research Project:** Leakage-aware malware detection via compact vision models and optimized CPU inference
**Status:** Phase 0 - Research/data access and project skeleton
**Last Updated:** 2024-11-22

---

## 📋 Overview

MalVisionNet is a research prototype that combines static PE file analysis with computer vision to detect Windows malware. Raw executable binaries are converted to grayscale or RGB byteplots, which are then classified using lightweight CNN/Transformer models trained on carefully audited datasets.

**Key Features:**
- ✅ Static-only analysis (no execution/sandboxing)
- ✅ Grayscale and RGB byte-to-image conversion
- ✅ Multiple vision model backbones (ResNet, MobileViT, LeViT)
- ✅ Leakage-aware evaluation (duplicate, family, temporal splits)
- ✅ ONNX CPU optimization with INT8 quantization
- ✅ Complete FastAPI backend and Next.js web interface

---

## 🎯 Research Questions

1. **Encoding:** Does RGB byte-triplet input provide measurable improvement over grayscale?
2. **Backbone:** Do lightweight transformers (MobileViT, LeViT) offer better accuracy/latency trade-offs than ResNet?
3. **Generalization:** How do family-held-out and temporal splits affect performance?
4. **Optimization:** Can ONNX INT8 quantization reduce model size and latency with acceptable accuracy loss?

---

## 📂 Project Structure

```
MalVisionNet/
├── configs/           # YAML configuration files
│   ├── data.yaml
│   ├── representation.yaml
│   ├── model.yaml
│   └── experiments/
├── data/              # Raw and processed datasets
│   ├── raw/
│   ├── manifests/
│   ├── interim/
│   └── processed/
├── src/malvision/     # Core Python modules
│   ├── data/          # Manifest generation, PE validation, splitting
│   ├── representations/ # Grayscale/RGB converters, resizing
│   ├── features/      # LightGBM baseline features
│   ├── models/        # ResNet18, MobileViT, LeViT implementations
│   ├── training/      # Training loops, losses, metrics
│   ├── evaluation/    # Metrics, robustness tests, error analysis
│   ├── explainability/# Grad-CAM, attention visualizations
│   ├── export/        # ONNX export, quantization, benchmarking
│   └── api/           # FastAPI endpoints
├── frontend/          # Next.js web application
│   ├── src/
│   │   ├── app/       # App Router pages and API routes
│   │   ├── components/ # Reusable UI components
│   │   ├── lib/       # API clients, utilities
│   │   └── types/     # TypeScript type definitions
├── artifacts/         # Model checkpoints and ONNX files
│   ├── checkpoints/
│   ├── onnx/
│   └── reports/
├── reports/           # Research reports and analysis
├── notebooks/         # Jupyter notebooks
└── tests/             # Unit tests
```

---

## 🚀 Quick Start

### Prerequisites

- **Python:** 3.11+
- **Node.js:** LTS version
- **Git:** For repository management
- **GPU:** Optional (for faster training), NVIDIA CUDA recommended

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/vixvo777/MalVisionNet.git
   cd MalVisionNet
   ```

2. **Create Python virtual environment**
   ```bash
   python -3.11 -m venv .venv
   .venv\Scripts\Activate.ps1  # Windows PowerShell
   # or
   source .venv/bin/activate   # Linux/Mac
   ```

3. **Install Python dependencies**
   ```bash
   pip install --upgrade pip wheel setuptools
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import sys, torch; print(f'Python: {sys.version}'); print(f'Torch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

5. **Build the frontend (optional)**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

---

## 📖 Dataset Access

### Malware Source: BODMAS
- **Dataset:** Blue Hexagon Open Dataset for Malware Analysis
- **Website:** https://whyisyoung.github.io/BODMAS/
- **Access:** University email required for original PE binaries
- **Status:** Phase 0 — Access request in progress

### Benign Source: Assemblage
- **Dataset:** Assemblage Windows PE corpus
- **Website:** https://assemblage-dataset.net/
- **License:** Per-binary license from source repositories
- **Status:** Previewing dataset card

### Comparison Datasets
- Malimg (historical grayscale benchmark)
- MaleVis (RGB image-only benchmark)
- EMBER (static-feature baseline)

---

## 🧪 Experiments

The project runs experiments in order (see `PLAN.md` for full details):

| Experiment | Description | Expected Outputs |
|------------|-------------|------------------|
| E0 | Data pipeline smoke test | Converter tests, manifest validation |
| E1 | Baseline | Grayscale + ResNet18 metrics |
| E2 | RGB evaluation | RGB + ResNet18 vs Grayscale comparison |
| E3 | Transformer comparison | MobileViT performance analysis |
| E4 | Width sensitivity | 128/256/512px width ablation |
| E5 | Resize interpolation | Area vs bilinear comparison |
| E6 | Generalization | Random vs temporal vs unseen-family splits |
| E7 | Optimization | ONNX FP32 vs INT8 benchmarking |
| E8 | Explainability | Grad-CAM/attention visualizations |

---

## 📊 Quick Reference

### Model Configurations

| Model | Input | Parameters | Notes |
|-------|-------|------------|-------|
| ResNet18 | 224×224 | ~11M | Stable baseline |
| MobileViT-XXS | 224×224 | ~4.6M | Lightweight transformer |
| LeViT-128S | 224×224 | ~7M | Efficiency-optimized transformer |

### Evaluation Metrics

- **Precision** — Positive predictive value
- **Recall (TPR)** — Malware detection rate
- **FPR** — Benign false positive rate
- **Macro-F1** — Balanced class F1 score
- **PR-AUC** — Precision-Recall AUC
- **ROC-AUC** — ROC curve AUC
- **Calibration** — Brier score, reliability plot

### Deployment

```bash
# Train a model
python src/malvision/training/train.py --config configs/experiments/e1_gray_resnet18.yaml

# Export to ONNX
python src/malvision/export/export_onnx.py --checkpoint artifacts/checkpoints/best_epoch.pt

# Run FastAPI server
uvicorn src.malvision.api.main:app --host 0.0.0.0 --port 8000

# Deploy with Docker Compose (Next.js + FastAPI)
docker-compose up --build
```

---

## ⚠️ Important Notes

### Safety First
- **Static-only analysis** — All files are read, never executed
- **No live malware** — Downloaded samples are analyzed without running
- **No third-party API calls** — Files are not sent to VirusTotal or other services

### Mitigation of Limitations
- **Packing/obfuscation:** Measured but not immune
- **Dataset source bias:** Addressed via repository-group splits and source-specific metrics
- **Zero-day claims:** Unseen-family testing only — not proof of new threats
- **Image resizing losses:** Quantified in experiments

### Reproducibility
- Use exact library versions from `requirements.txt`
- Freeze dataset manifests and split hashes
- Report hardware specs, seeds, and training logs
- All models trained with fixed random seeds

---

## 📚 Research References

See `PLAN.md` for the full literature review with 16 citations including:
- Nataraj et al. (2011) — Malware image visualization foundation
- Nguyen et al. (2023) — RGB trigram encoding study
- Yang et al. (2021) — BODMAS temporal dataset
- Mehta & Rastegari (2022) — MobileViT architecture
- Liu et al. (2024) — Assemblage dataset construction
- Larson & Leach (2026) — Leakage and duplication audit

---

## 🤝 Contributing

Currently in **research phase** — not open to external contributions yet. The project follows strict research protocols and Git attribution requirements.

---

## 📝 License and Data Use

- **Data licenses:** Each sample follows the license of its source repository
- **BODMAS:** Academic research use only — requires institutional affiliation
- **Assemblage:** Per-binary terms from source licenses
- **Repo:** This code repository follows MIT-style licensing

---

## 📞 Contact

- **GitHub:** https://github.com/vixvo777/MalVisionNet
- **Issue Tracker:** For bug reports or documentation improvements
- **Research inquiries:** See dataset access in PLAN.md

---

## 🗺️ Project Roadmap

| Phase | Status | Completion Date |
|-------|--------|-----------------|
| Phase 0 | ✅ In Progress | 2024-11-22 |
| Phase 1 | ⏳ Pending | |
| Phase 2 | ⏳ Pending | |
| Phase 3 | ⏳ Pending | |
| Phase 4 | ⏳ Pending | |
| Phase 5 | ⏳ Pending | |
| Phase 6 | ⏳ Pending | |
| Phase 7 | ⏳ Pending | |
| Phase 8 | ⏳ Pending | |
| Phase 9 | ⏳ Pending | |

**Full Phase Breakdown:** See `PLAN.md` Sec. 14

---

*This is a research prototype. Malware classification results are research scores, not guaranteed security verdicts.*