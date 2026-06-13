<h1 align="center">🛡️ Tagalog NSFW Text Classifier</h1>

<p align="center">
  <strong>LSTM</strong> &amp; <strong>BiLSTM</strong> deep-learning models for detecting NSFW (Not Safe For Work)
  content in <strong>Tagalog / Taglish</strong> (Filipino-English) text.
</p>

<p align="center">
  <img alt="Python"     src="https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white">
  <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white">
  <img alt="Tuner"      src="https://img.shields.io/badge/HPO-Keras%20Tuner-D00000?logo=keras&logoColor=white">
  <img alt="Models"     src="https://img.shields.io/badge/Models-LSTM%20%7C%20BiLSTM-4B8BBE">
  <img alt="Task"       src="https://img.shields.io/badge/Task-Binary%20Classification-success">
  <img alt="Deploy"     src="https://img.shields.io/badge/Export-TFLite-blueviolet">
</p>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Models](#-models)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Outputs & Artifacts](#-outputs--artifacts)
- [License](#-license)
- [Credits](#-credits)

## 🔍 Overview

This project trains binary text classifiers that flag content as either **`safe`** or
**`nsfw`**. It is tailored for the **Filipino language context**, where messages frequently mix
Tagalog and English (*Taglish*) and rely on local slang — cases that off-the-shelf English-only
moderation models tend to miss.

The study is structured as a **2×2 comparison** — two architectures (**LSTM** and **BiLSTM**),
each in a **baseline** variant (hand-picked hyperparameters) and a **hyperparameter-optimized**
variant (Keras Tuner `RandomSearch`):

|              | Baseline (fixed HPs)        | Hyperparameter-optimized        |
| ------------ | --------------------------- | ------------------------------- |
| **LSTM**     | `NLP/LSTM.ipynb`            | `NLP/LSTM_hyperparameters.ipynb`   |
| **BiLSTM**   | `NLP/BiLSTM.ipynb`          | `NLP/BiLSTM_hyperparameters.ipynb` |

**Key features:**

- 🌐 **Taglish-aware** — trained on real-world Filipino social-media text.
- 🧹 **Obfuscation-resistant preprocessing** — leetspeak, emoji, and inserted-noise normalization.
- 🎚️ **Calibrated decision threshold** — F1-optimized on the validation set (only adopted over the neutral 0.50 when the gain is meaningful).
- 📦 **Deployment-ready** export to both Keras and **TensorFlow Lite** formats.
- 📊 Rich evaluation: classification report, confusion matrix, F1 / AUROC / AUPRC, and training-history plots.

## 🤖 Models

All four notebooks share the same backbone (embedding → single recurrent layer → dense head) and the
same **70 / 15 / 15** train / val / test split. They differ in architecture and how hyperparameters
are chosen:

| | Baseline (`LSTM` / `BiLSTM`) | Hyperparameter (`*_hyperparameters`) |
| --- | --- | --- |
| **Recurrent layer** | Single LSTM / Bidirectional LSTM | Single LSTM / Bidirectional LSTM |
| **Hyperparameter tuning** | — | ✅ Keras Tuner `RandomSearch` (10 trials) |
| **Tuned params** | — | `dropout`, `lr`, `weight_decay`, `beta_1`, `batch_size` |
| **Loss** | Binary cross-entropy (label smoothing 0.1) | Binary cross-entropy (label smoothing 0.1) |
| **Optimizer** | Adam | AdamW (tuned `lr` / `weight_decay` / `beta_1`) |
| **Threshold tuning** | ✅ Max-F1 on validation | ✅ Max-F1 on validation |
| **Target runtime** | Local / Jupyter / Colab | Google Colab (GPU) |

## 📈 Results

Final metrics on the held-out **test set** (n = 837; 462 `safe` / 375 `nsfw`), decision threshold
**0.50** for every model:

| Model | Accuracy | Macro-F1 | NSFW-F1 | NSFW recall | ROC-AUC | PR-AUC |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LSTM — baseline** 🏆 | **0.8973** | **0.8955** | **0.8819** | **0.8560** | **0.9403** | **0.9507** |
| BiLSTM — HPO | 0.8925 | 0.8894 | 0.8711 | 0.8107 | 0.9390 | 0.9495 |
| LSTM — HPO | 0.8853 | 0.8833 | 0.8681 | 0.8427 | 0.9340 | 0.9470 |
| BiLSTM — baseline | 0.8805 | 0.8781 | 0.8607 | 0.8240 | 0.9370 | 0.9469 |

**Takeaways:**

- The **plain LSTM baseline wins on every metric** — on this dataset, neither bidirectionality nor
  hyperparameter search beat well-chosen defaults.
- HPO **hurt** the already-well-tuned LSTM (−1.2 pts accuracy) but **helped** the under-regularized
  BiLSTM (+1.2 pts), since the search added dropout / weight decay the BiLSTM baseline lacked.
- All models overfit the **3,905-sample** training set (train accuracy ≈ 0.99 by epoch 3–5); the data
  size — not the architecture — is the binding constraint.
- The BiLSTM-HPO model is the most **precise** (highest NSFW precision) but least **sensitive**
  (lowest NSFW recall) — a different operating point rather than strictly better or worse.

> Metrics are read from each model's `test_metrics.csv`. Re-running the notebooks may shift the last
> digits slightly due to nondeterministic GPU training.

## 📂 Project Structure

```
THESIS_2/
├── README.md                            # You are here
└── NLP/
    ├── LSTM.ipynb                       # Baseline unidirectional LSTM
    ├── LSTM_hyperparameters.ipynb       # LSTM + Keras Tuner HPO
    ├── BiLSTM.ipynb                     # Baseline bidirectional LSTM
    ├── BiLSTM_hyperparameters.ipynb     # BiLSTM + Keras Tuner HPO
    ├── cleaned_data.csv                 # Preprocessed dataset  (text, label)
    ├── nsfw_dataset_text.csv            # Raw labelled dataset   (language, category, text_clean, type)
    ├── models/                          # Trained artifacts (one dir per model)
    │   ├── lstm_tagalog_nsfw/
    │   ├── lstm_hyperparameter_tagalog_nsfw/
    │   ├── bilstm_tagalog_nsfw/
    │   └── bilstm_hyperparameter_tagalog_nsfw/
    └── tuning_results/                  # Keras Tuner studies + trial CSVs
        ├── lstm_nsfw_tagalog_v2/        # (hpo_trials.csv)
        └── bilstm_nsfw_tagalog_v2/      # (hpo_trials_bilstm.csv)
```

## 🗂️ Dataset

All models train on `cleaned_data.csv`, a two-column dataset:

| Column | Description |
| --- | --- |
| `text` | A short Tagalog/Taglish message. |
| `label` | Class label — `safe` or `nsfw`. |

The raw source (`nsfw_dataset_text.csv`) additionally carries `language` and `category` annotations
that were collapsed during cleaning. The cleaned set is **5,579 rows** and roughly balanced
(**3,079 `safe` / 2,500 `nsfw`**); because of this balance, data augmentation is disabled by default.

> ⚠️ **Content warning:** the dataset contains explicit language by design, as it is required to
> train a moderation model. It is intended strictly for research and content-safety purposes.

## 🏗️ Model Architecture

Each model is an embedding layer feeding a **single** recurrent layer and a dense classification head.
The variants differ only in the recurrent direction and (for the HPO notebooks) the tuned optimizer.

**Unidirectional — `LSTM.ipynb` / `LSTM_hyperparameters.ipynb`:**

```
Embedding → SpatialDropout1D → LSTM → Dense → Dropout → Dense → Sigmoid (1 unit)
```

**Bidirectional — `BiLSTM.ipynb` / `BiLSTM_hyperparameters.ipynb`:**

```
Embedding → SpatialDropout1D → Bidirectional(LSTM) → Dense → Dropout → Dense → Sigmoid (1 unit)
```

- **Embedding** — learns dense word vectors from the tokenized vocabulary (`mask_zero=True` so padding is ignored).
- **(Bi)LSTM** — the baseline reads the sequence forward only; the BiLSTM reads it in both directions.
- **Dense + Dropout** — adds capacity while regularizing (L2 on the dense layer) against overfitting.
- **Sigmoid output** — produces a single NSFW probability for binary classification.

Shared defaults: `vocab_size=10000`, `embedding_dim=64`, recurrent/dense units `=64`,
`max_sequence_length=64`.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- (Optional) An NVIDIA GPU with CUDA for accelerated training — the notebooks detect and use it automatically.

### Installation

```bash
git clone <your-repo-url>
cd THESIS_2

# Core dependencies (all notebooks)
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn

# Extra dependencies for preprocessing and the hyperparameter notebooks
pip install keras-tuner unidecode emoji
```

> 💡 The `*_hyperparameters.ipynb` notebooks are authored for **Google Colab** (they use `!pip install`
> and a `google.colab` download cell) — drop or guard that final cell when running locally.

## 💻 Usage

Open whichever notebook you want to run:

```bash
jupyter notebook NLP/LSTM.ipynb                       # baseline LSTM
jupyter notebook NLP/LSTM_hyperparameters.ipynb       # LSTM + HPO
jupyter notebook NLP/BiLSTM.ipynb                     # baseline BiLSTM
jupyter notebook NLP/BiLSTM_hyperparameters.ipynb     # BiLSTM + HPO
```

All four follow the same high-level workflow, orchestrated by the `ModelTrainer` class
(`trainer.run_all()`):

1. **Load** the dataset (`load_dataset`).
2. **Preprocess** — normalize, tokenize, pad, and split into train/val/test sets (`preprocess_data`).
3. **Build** the model (`build_lstm_model` / `build_bilstm_model`), or run the **Keras Tuner search** (HPO notebooks).
4. **Train** with checkpointing and early stopping (`train_model`).
5. **Evaluate** — calibrate the threshold and compute metrics on validation, then the held-out test set (`evaluate_model`).
6. **Export** to Keras + TFLite artifacts (`save_artifacts`, `export_to_tflite`).

### Quick inference

After training, classify new text with the built-in helper (point the paths at any of the four model dirs):

```python
test_inference(
    model_path="models/lstm_tagalog_nsfw/final_model.keras",
    tokenizer_path="models/lstm_tagalog_nsfw/tokenizer.pickle",
    config_path="models/lstm_tagalog_nsfw/config.json",
    test_texts=["Magandang umaga sa lahat", "Kumusta ka"],
)
```

## 📦 Outputs & Artifacts

Each notebook writes to its own directory under `models/` (e.g. `models/lstm_tagalog_nsfw/`):

| Artifact | Purpose |
| --- | --- |
| `final_model.keras` / `best_model.keras` | Trained Keras model (final + best checkpoint). |
| `model.tflite` | Lightweight model for on-device / mobile deployment. |
| `tokenizer.pickle` | Fitted tokenizer for reproducing the input encoding. |
| `config.json` | Saved hyperparameters and decision threshold. |
| `validation_metrics.csv` / `test_metrics.csv` | Headline metrics per split. |
| `*_classification_report.csv` / `*_predictions.csv` | Per-class report and per-sample predictions. |
| `*_confusion_matrix.png` | Confusion matrix. |
| `training_history.png` / `.csv` | Accuracy / loss curves over epochs. |
| `inference_results.csv` / `training.log` | Sample-inference dump and full run log. |

The hyperparameter notebooks additionally persist the Keras Tuner study under
`tuning_results/<study>/` together with a ranked trial summary
(`hpo_trials.csv` for the LSTM study, `hpo_trials_bilstm.csv` for the BiLSTM study).

## 📄 License

This project is part of an academic thesis. Please contact the author before reuse or redistribution.

## 🙌 Credits

README design and structure adapted from the [**awesome-readme**](https://github.com/matiassingers/awesome-readme)
collection — specifically the centered-title-with-badges-and-section-links style of the
[**doomemacs**](https://github.com/doomemacs/doomemacs#readme) README.
