# Real-Time Multi-Class Activity Recognition (Classical ML vs. 1D-CNN)

![Live Demo](docs/demo.gif)

## 🚀 Overview

This project presents an **end-to-end real-time activity recognition system** that classifies human actions (throwing, drinking, driving) from sensor data streams.

It integrates two complementary approaches:
- A **classical Random Forest pipeline** using engineered statistical features.
- A **deep 1D-Convolutional Neural Network (1D-CNN)** trained directly on raw sequential data.

A unified inference engine allows direct runtime comparison between both models, offering insights into trade-offs between **classical machine learning** and **deep learning** approaches.  
The system is modular, containerized, and includes automated testing and CI/CD pipelines for full reproducibility.

---

## ✨ Key Features

- **Unified Multi-Model Engine:** Easily switch between `RandomForest` and `1D-CNN` for real-time predictions using a single CLI flag (`--model-type`).
- **Deep Learning Workflow:** Complete PyTorch training pipeline for sequential 1D sensor data.
- **Classical ML Workflow:** Scikit-learn training pipeline with automatic feature extraction (`mean`, `std`, `rms`, etc.).
- **Command-Line Interface (CLI):** Built with **Typer** and **Rich**, providing intuitive commands (`train`, `train-dl`, `predict-stream`).
- **Containerized Environment:** Deploy anywhere using a lightweight **Dockerfile**.
- **Automated Quality Control:**
  - **GitLab CI/CD:** Runs linting (`flake8`) and unit tests (`pytest`) on each commit.
  - **Pre-Commit Hooks:** Enforces consistent formatting (`black`) and static analysis.
- **Dependency Management:** Managed with **Poetry** for reproducible builds and isolation.

---

## 🛠️ Technology Stack

| Category | Tools & Libraries |
|-----------|-------------------|
| **Language** | Python 3.10+ |
| **ML / DL** | Scikit-learn, PyTorch, Pandas, NumPy |
| **CLI** | Typer, Rich |
| **DevOps** | Docker, Poetry, Pre-commit, GitLab CI |
| **Testing & Quality** | Pytest, Pytest-Mock, Black, Flake8 |
| **Visualization** | Matplotlib |

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SoroushHaghi/Job-Prep2025.git
   cd Job-Prep2025
   ```

2. **Install Poetry:**  
   Follow the [official Poetry installation guide](https://python-poetry.org/docs/#installation).

3. **Install dependencies:**
   ```bash
   poetry install
   ```

4. **Activate pre-commit hooks:**
   ```bash
   poetry run pre-commit install
   ```

---

## ▶️ Usage

All commands must be executed inside the Poetry environment using `poetry run`.

### 1. Build Feature Dataset (for RandomForest)
Generate engineered features for the classical ML model:
```bash
poetry run python -m src.main build-features
```

### 2. Train Models
Train one or both available models:
```bash
# RandomForest model
poetry run python -m src.main train

# 1D-CNN model
poetry run python -m src.main train-dl
```

### 3. Run Real-Time Inference Simulation
Use pre-recorded sensor data to simulate streaming predictions:
```bash
# Default (RandomForest)
poetry run python -m src.main predict-stream --data-file data/throwing.csv

# 1D-CNN model
poetry run python -m src.main predict-stream --data-file data/throwing.csv --model-type cnn --model-file models/cnn_model.pth
```

### 4. Run via Docker
Build and execute the system inside a Docker container:
```bash
# Build image
docker build -t activity-recognition .

# Run inference
docker run --rm activity-recognition predict-stream --data-file data/drinking.csv
docker run --rm activity-recognition predict-stream --data-file data/drinking.csv --model-type cnn --model-file models/cnn_model.pth
```

---

## 📊 Model Comparison

Both models were trained on the same dataset to compare generalization and inference performance.

| Metric | RandomForest | 1D-CNN |
|:--|:--:|:--:|
| **Overall Accuracy** | 91.7 % | **93.0 %** |
| **Throwing (F1)** | 0.94 | **0.96** |
| **Drinking (F1)** | 0.89 | **0.90** |
| **Driving (F1)** | 0.92 | 0.92 |

---

## 🗂️ Project Structure

```
├── data/               # Raw data & generated features
├── docs/               # Docs and visualization assets (e.g. confusion_matrix.png)
├── models/             # Saved trained models (.joblib, .pth)
├── src/                # Source code
│   ├── app.py              # Core processing logic
│   ├── dataset_builder.py  # Builds features.csv
│   ├── drivers.py          # Sensor simulation drivers
│   ├── features.py         # Feature extraction functions
│   ├── main.py             # CLI entry point
│   ├── model.py            # 1D-CNN model definition
│   ├── model_trainer.py    # RandomForest training & evaluation
│   ├── predictor.py        # Multi-model inference engine
│   ├── train_dl.py         # CNN training & evaluation
│   └── utils.py            # Helper utilities
├── tests/              # Unit tests
├── .gitlab-ci.yml      # CI/CD configuration
├── .pre-commit-config.yaml # Pre-commit setup
├── Dockerfile          # Container build file
├── pyproject.toml      # Poetry project metadata
└── README.md           # Documentation
```

---

## 📘 About

This project demonstrates the **design, training, and deployment** of a hybrid machine learning system, highlighting practical trade-offs between **classical feature-based** models and **deep neural architectures**.  
It emphasizes:
- **Real-time inference capability**
- **Reproducible ML workflows**
- **Modern Python packaging and CI/CD best practices**

---

**Author:** [Soroush Haghi](https://github.com/SoroushHaghi)  
**Last Updated:** October 2025
