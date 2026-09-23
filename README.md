<h1 align="center">🛡️ NSFW Detection for the Filipino Context</h1>

<p align="center">
  A two-track thesis benchmark: <strong>NLP text classifiers</strong> for <strong>Tagalog / Taglish</strong>
  (Filipino-English) NSFW text, and <strong>Computer-Vision object detectors</strong> for NSFW imagery.
</p>

<p align="center">
  <img alt="Python"       src="https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white">
  <img alt="TensorFlow"   src="https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white">
  <img alt="PyTorch"      src="https://img.shields.io/badge/PyTorch-Transformers-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="Ultralytics"  src="https://img.shields.io/badge/Ultralytics-YOLO-111F68">
  <img alt="Task"         src="https://img.shields.io/badge/Tasks-Text%20Classification%20%7C%20Object%20Detection-success">
  <img alt="Deploy"       src="https://img.shields.io/badge/Export-TFLite-blueviolet">
</p>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [NLP Track — Text Classification](#-nlp-track--text-classification)
  - [Models](#nlp-models)
  - [Dataset](#nlp-dataset)
  - [Architecture](#nlp-architecture)
  - [Results](#nlp-results)
- [Computer-Vision Track — Object Detection](#-computer-vision-track--object-detection)
  - [Models](#cv-models)
  - [Dataset](#cv-dataset)
  - [Pipeline](#cv-pipeline)
  - [Results](#cv-results)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Outputs & Artifacts](#-outputs--artifacts)
- [License](#-license)
- [Credits](#-credits)

## 🔍 Overview

This thesis benchmarks a broad set of models for **NSFW (Not Safe For Work) detection** tailored to the
**Filipino context**, split into two independent tracks:

- **NLP track** — binary **`safe` / `nsfw`** text classifiers for Tagalog/Taglish, where messages
  frequently mix Tagalog and English and rely on local slang that off-the-shelf English-only
  moderation models tend to miss.
- **Computer-Vision track** — **object detectors** trained on a Roboflow *erotica-detection* dataset,
  spanning several architecture families to compare accuracy vs. on-device efficiency.

Every model is evaluated in a **baseline** variant (hand-picked hyperparameters) and a
**hyperparameter-optimized (HPO)** variant, so each architecture appears as a matched pair of
notebooks (`<Model>.ipynb` and `<Model>_hyperparameters.ipynb`). The overarching goal is a
**deployment-oriented** comparison: most models export to **TensorFlow Lite** for edge inference.

> ⚠️ **Content warning:** both datasets contain explicit language / imagery by design, as required to
> train moderation models. They are intended strictly for research and content-safety purposes.

## 📂 Project Structure

```
THESIS_2/
├── README.md
├── .gitignore
├── NLP/                                    # Text-classification track
│   ├── LSTM.ipynb / LSTM_hyperparameters.ipynb
│   ├── BiLSTM.ipynb / BiLSTM_hyperparameters.ipynb
│   ├── RoBERTa_Tagalog.ipynb / …_hyperparameters.ipynb
│   ├── DOST_RoBERTa.ipynb / …_hyperparameters.ipynb
│   ├── DistilBert_Tagalog.ipynb / …_hyperparameters.ipynb
│   ├── MobileBERT.ipynb / …_hyperparameters.ipynb
│   ├── TinyBERT.ipynb / …_hyperparameters.ipynb
│   ├── Multilingual_MiniLM.ipynb / …_hyperparameters.ipynb
│   ├── cleaned_data.csv                    # Preprocessed dataset  (text, label)
│   ├── nsfw_dataset_text.csv               # Raw labelled dataset  (language, category, text_clean, type)
│   ├── models/                             # Trained artifacts (one dir per model)
│   └── tuning_results/                     # Keras Tuner studies + trial CSVs
└── Computer_Vision/                        # Object-detection track
    ├── yolov5{n,s}-{320,640}.ipynb         # YOLOv5 (repo-based)
    ├── yolov10n.ipynb / yolov10n(1).ipynb  # YOLOv10 (base + tuned)
    ├── yolov11{n,s}[-{320,640,768}].ipynb  # YOLOv11 scale/resolution sweep
    ├── yolov12n[-{320,640}].ipynb          # YOLOv12
    ├── yolov26n.ipynb                      # YOLOv26
    ├── rf_detr_nano.ipynb / …_hyperparameters.ipynb
    ├── efficientdet.ipynb / …_hyperparameters.ipynb
    ├── mobilenetssd_torch.ipynb / …_hyperparameters.ipynb
    └── results/                            # Metrics + plots per detector (one dir per model)
```

> Datasets, archives (`*.zip`), virtualenvs, secrets, and WSL `*:Zone.Identifier` cruft are
> git-ignored (see [.gitignore](.gitignore)). Trained models and result CSVs are **kept in-repo** as
> thesis outputs.

## 🗣️ NLP Track — Text Classification

<a id="nlp-models"></a>

### Models

Eight architectures, each with a baseline and a hyperparameter-optimized notebook. The recurrent
models are trained from scratch; the rest fine-tune pretrained (mostly Tagalog-aware) transformers:

| Family | Notebook stem | Pretrained checkpoint |
| --- | --- | --- |
| **LSTM** (from scratch) | `LSTM` | — |
| **BiLSTM** (from scratch) | `BiLSTM` | — |
| **RoBERTa Tagalog** | `RoBERTa_Tagalog` | `danjohnvelasco/roberta-tagalog-base-cohfie-v1` |
| **DOST RoBERTa** | `DOST_RoBERTa` | `dost-asti/RoBERTa-tl-sentiment-analysis` |
| **DistilBERT Tagalog** | `DistilBert_Tagalog` | `jcblaise/distilbert-tagalog-base-cased` |
| **MobileBERT** | `MobileBERT` | `google/mobilebert-uncased` |
| **TinyBERT** | `TinyBERT` | `huawei-noah/TinyBERT_General_4L_312D` |
| **Multilingual MiniLM** | `Multilingual_MiniLM` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |

All notebooks share the same `cleaned_data.csv`, the same **70 / 15 / 15** train/val/test split, an
F1-optimized decision threshold calibrated on validation, and **TFLite** export for deployment.
The exception is **MobileBERT**, which splits **70 / 25 / 5** (test n = 280), so its numbers are not
directly comparable.

**Key features:**

- 🌐 **Taglish-aware** — trained on real-world Filipino social-media text.
- 🧹 **Obfuscation-resistant preprocessing** — leetspeak, emoji, and inserted-noise normalization.
- 🎚️ **Calibrated decision threshold** — F1-optimized on validation (adopted over the neutral 0.50 only when the gain is meaningful).
- 📦 **Deployment-ready** export to Keras / PyTorch and **TensorFlow Lite**.
- 📊 Rich evaluation: classification report, confusion matrix, F1 / AUROC / AUPRC, training-history plots.

<a id="nlp-dataset"></a>

### Dataset

Models train on `cleaned_data.csv`, a two-column dataset:

| Column | Description |
| --- | --- |
| `text` | A short Tagalog/Taglish message. |
| `label` | Class label — `safe` or `nsfw`. |

The raw source (`nsfw_dataset_text.csv`) additionally carries `language` and `category` annotations
that were collapsed during cleaning. The cleaned set is **5,579 rows** and roughly balanced
(**3,079 `safe` / 2,500 `nsfw`**); because of this balance, data augmentation is disabled by default.

<a id="nlp-architecture"></a>

### Architecture

The recurrent models are an embedding layer feeding a **single** recurrent layer and a dense head:

```
LSTM   : Embedding → SpatialDropout1D → LSTM               → Dense → Dropout → Dense → Sigmoid
BiLSTM : Embedding → SpatialDropout1D → Bidirectional(LSTM) → Dense → Dropout → Dense → Sigmoid
```

Shared defaults: `vocab_size=10000`, `embedding_dim=64`, recurrent/dense units `=64`,
`max_sequence_length=64`, `mask_zero=True`. The transformer notebooks instead fine-tune their
pretrained checkpoint with a binary classification head.

<a id="nlp-results"></a>

### Results

Verified metrics for the recurrent baselines on the held-out **test set** (n = 837; 462 `safe` /
375 `nsfw`), decision threshold **0.50** (Accuracy and Macro-F1 read directly from notebook output;
per-class and AUC metrics are saved to each model's `*_test_metrics.csv`):

| Model | Accuracy | Macro-F1 |
| --- | :---: | :---: |
| **BiLSTM — HPO** 🏆 | **0.9020** | **0.8999** |
| LSTM — baseline | 0.8984 | 0.8967 |
| LSTM — HPO | 0.8937 | 0.8918 |
| BiLSTM — baseline | 0.8853 | 0.8837 |

Transformer results on the same test set (n = 837), threshold **0.50**. Metric files live in each
model's `NLP/models/<model>/` directory; the DOST-RoBERTa HPO row is read from notebook output:

| Model | Accuracy | Macro-F1 |
| --- | :---: | :---: |
| **Multilingual MiniLM — HPO** 🏆 | **0.9188** | **0.9175** |
| Multilingual MiniLM — baseline | 0.9140 | 0.9124 |
| DOST RoBERTa — HPO | 0.9128 | 0.9110 |
| DOST RoBERTa — baseline | 0.8961 | 0.8930 |
| TinyBERT — baseline | 0.8853 | 0.8829 |
| TinyBERT — HPO | 0.8757 | 0.8736 |
| MobileBERT — baseline ¹ | 0.8893 | 0.8869 |
| MobileBERT — HPO ¹ | 0.5571 | 0.3721 |

¹ Different split (test n = 280, 155 `safe` / 125 `nsfw`). The HPO run collapsed to predicting
almost everything as `safe` (NSFW recall 0.02).

RoBERTa Tagalog and DistilBERT Tagalog have **no results yet**: the RoBERTa Tagalog notebooks have not
been run, and the only DistilBERT Tagalog run was interrupted during model download.

**Takeaways (recurrent models):** **BiLSTM — HPO leads**, showing that bidirectionality and
hyperparameter search together can outperform simple defaults. The plain LSTM baseline trails closely
in second. All models overfit the ~3,900-sample training set (train accuracy ≈ 0.99 by epoch 3–5);
the data size, not the architecture, remains the binding constraint.

**Takeaways (transformers):** **Multilingual MiniLM — HPO** is the best NLP model overall, just ahead
of DOST RoBERTa — HPO and about 1.7 points above the best recurrent model. Hyperparameter search
helped MiniLM and DOST RoBERTa but *hurt* TinyBERT and broke MobileBERT, so tuned runs are not
automatically better than baselines.

> Re-running notebooks may shift the last digits due to nondeterministic GPU training.

## 🖼️ Computer-Vision Track — Object Detection

<a id="cv-models"></a>

### Models

Multiple detector families are benchmarked, most across scale (`n`/`s`) and input resolution
(`320`/`640`/`768`). Baseline and hyperparameter-optimized variants are provided where applicable:

| Family | Notebooks | Framework |
| --- | --- | --- |
| **YOLOv5** | `yolov5{n,s}-{320,640}` (4) | cloned `ultralytics/yolov5` repo |
| **YOLOv10** | `yolov10n`, `yolov10n(1)` (tuned) | `THU-MIG/yolov10` (`ultralytics` fork, `YOLOv10`) |
| **YOLOv11** | `yolov11n`, `yolov11s`, `yolov11{n-320,n-768,s-640,s_640,s_320}` | `ultralytics` |
| **YOLOv12** | `yolov12n`, `yolov12n-320`, `yolov12n-640` | `ultralytics` |
| **YOLOv26** | `yolov26n` | `ultralytics` (latest, YOLO26 support) |
| **RF-DETR (nano)** | `rf_detr_nano` (+ `_hyperparameters`) | `rfdetr==1.2.1` |
| **EfficientDet** | `efficientdet` (+ `_hyperparameters`) | PyTorch / torchvision |
| **MobileNet-SSD** | `mobilenetssd_torch` (+ `_hyperparameters`) | PyTorch |

The YOLOv5 notebooks, `rf_detr_nano_hyperparameters`, and `mobilenetssd_torch_hyperparameters` have
no saved outputs yet (not run).

<a id="cv-dataset"></a>

### Dataset

The detectors train on the Roboflow **`erotica-detection`** project (workspace `renaire-odarve`),
downloaded per-notebook via the Roboflow API in the format each framework expects (`yolov5`,
`yolov11`, COCO, etc.). It has three classes: **`Gore`**, **`nsfw`**, and **`safe`** (the validation
split used for EfficientDet has 2,416 images / 3,954 boxes). Dataset archives are git-ignored and
re-downloaded on demand.

> ⚠️ The training notebooks reference specific Roboflow dataset **versions** and pull a
> `ROBOFLOW_API_KEY` from Colab secrets. Pin a single dataset version across models when comparing
> results, and evaluate on the **test** split (not the default `val` split) for held-out numbers.

<a id="cv-pipeline"></a>

### Pipeline

Each notebook follows: install deps → download the dataset → train → visualize curves / confusion
matrix → evaluate → run sample inference → export and zip artifacts. The YOLO notebooks export to **TFLite**;
EfficientDet, MobileNet-SSD, and RF-DETR export to **ONNX**. The notebooks are
authored for **Google Colab (GPU)**.

<a id="cv-results"></a>

### Results

Metrics extracted to `Computer_Vision/results/<model>/`. YOLO rows are the best epoch (highest
mAP@.5:.95) from `results.csv`, out of 150 epochs:

| Model | Split | Precision | Recall | mAP@.5 | mAP@.5:.95 |
| --- | :---: | :---: | :---: | :---: | :---: |
| **YOLOv12n** 🏆 | val | 0.764 | 0.717 | **0.753** | **0.376** |
| YOLOv26n | val | 0.762 | 0.703 | 0.742 | 0.371 |
| EfficientDet-D0 — baseline | val | 0.739 | 0.671 | 0.659 | 0.286 |
| EfficientDet-D0 — HPO | val | 0.736 | 0.667 | 0.644 | 0.283 |
| YOLOv11n | val | 0.641 | 0.576 | 0.598 | 0.303 |
| MobileNet-SSD (v3-large) | test | — | — | 0.566 | 0.247 |

> ⚠️ These rows are not a clean comparison: most are **validation** numbers (the EfficientDet test
> split was empty), and the notebooks may use different Roboflow dataset versions.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- (Optional) An NVIDIA GPU with CUDA — the notebooks detect and use it automatically.
- Most notebooks are authored for **Google Colab**; the CV notebooks additionally need a Roboflow
  API key stored as the Colab secret `ROBOFLOW_API_KEY`.

### Installation

```bash
git clone <your-repo-url>
cd THESIS_2

# NLP — recurrent notebooks
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn keras-tuner unidecode emoji

# NLP — transformer notebooks (HPO variants also need optuna)
pip install torch transformers accelerate datasets optuna
pip install onnxscript "optimum[onnxruntime]" onnxconverter-common litert-torch torchao

# Computer Vision — YOLO (YOLOv10 installs git+https://github.com/THU-MIG/yolov10.git instead)
pip install ultralytics supervision roboflow onnx onnx2tf

# Computer Vision — EfficientDet / MobileNet-SSD / RF-DETR
pip install albumentations torchmetrics onnx onnxruntime onnxscript optuna "rfdetr==1.2.1"
```

> 💡 The `*_hyperparameters.ipynb` and CV notebooks use `!pip install` and a `google.colab` download
> cell — drop or guard those cells when running locally.

## 💻 Usage

Open whichever notebook you want to run, e.g.:

```bash
jupyter notebook NLP/LSTM.ipynb                        # baseline LSTM
jupyter notebook NLP/DOST_RoBERTa_hyperparameters.ipynb # tuned transformer
jupyter notebook Computer_Vision/yolov5n-320.ipynb      # YOLOv5-nano @ 320px
```

The NLP notebooks follow a shared workflow orchestrated by a `ModelTrainer` class
(`trainer.run_all()`): **load → preprocess → build/tune → train → evaluate → export**.

### Quick inference (NLP recurrent models)

```python
test_inference(
    model_path="models/lstm_tagalog_nsfw/final_model.keras",
    tokenizer_path="models/lstm_tagalog_nsfw/tokenizer.pickle",
    config_path="models/lstm_tagalog_nsfw/config.json",
    test_texts=["Magandang umaga sa lahat", "Kumusta ka"],
)
```

## 📦 Outputs & Artifacts

Each **NLP** notebook writes to its own directory under `NLP/models/`:

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

The recurrent HPO notebooks additionally persist their Keras Tuner study under `NLP/tuning_results/<study>/`
with a ranked trial summary (`hpo_trials*.csv`). The transformer HPO notebooks use **Optuna** instead
and keep their trials inside their own model directory
(`<model>_hyperparameters/tuning_results/hpo_trials_*.csv` and `models/*_hpo_results.json`).

The **transformer** directories (`dost_roberta/`, `multilingual_minilm*/`, `tinybert*/`,
`mobilebert*/`) hold metrics, reports, plots, and deployment configs only. Their weights and
TFLite/ONNX exports are too large for the repo and stay in the Colab download zips.

Each **CV** notebook produces an Ultralytics/framework `runs/` directory (weights, curves, confusion
matrix), a TFLite export, and a zipped `*_results.zip` / `*_saved_model.zip` for download. The metrics
and plots from those zips are kept in `Computer_Vision/results/<model>/`.

## 📄 License

This project is part of an academic thesis. Please contact the author before reuse or redistribution.

## 🙌 Credits

README design adapted from the [**awesome-readme**](https://github.com/matiassingers/awesome-readme)
collection — the centered-title-with-badges-and-section-links style of the
[**doomemacs**](https://github.com/doomemacs/doomemacs#readme) README.
