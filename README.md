# Toxic Comment Classification Challenge 🧪

This repository contains a solution in progress for the Jigsaw Toxic Comment Classification Challenge, as part of a technical interview process.

The task is to build a multilabel classification model capable of detecting various types of toxicity in Wikipedia comments.

## 📦 Project Overview

1. Training
  Run `main.py` to train a new model. It will:
    - Load and preprocess data from `data/`
    - Train a model pipeline (e.g., TF-IDF + Logistic Regression)
    - Evaluate performance on a validation set
    - Save the trained model and vectorizer under `models/<model_version>/`

2. Serving Predictions
  Use the FastAPI app in `src/api.py` to serve your model:
    - The API loads the requested model version at startup
    - Accepts comment text via a `/predict` POST endpoint
    - Returns toxicity predictions for the comment

3. Experimentation
  Use notebooks in `notebooks/` for exploratory data analysis, feature engineering, or testing new model ideas.

## 📁 Project structure

```bash
toxicity-classification/
│
├── data/               # Input data (train.csv, test.csv, etc.)
├── notebooks/          # Exploratory notebooks (EDA, modeling)
├── src/
│   ├── main.py          # FastAPI endpoints (for inference)
│   ├── model.py         # Class-based model wrapper (load, predict)
│   ├── config.py        # Paths, label list, constants
│   └── utils.py         # Utility functions (cleaning, evaluation, etc.)
├── .venv/              # Virtual environment (not committed)
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
├── setup.py            # For editable install of src/
└── .gitignore
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/toxicity-classification.git
cd toxicity-classification
```

2. Create a virtual environment:
```bash
python3 -m venv .venv
```
3. Activate the environment:
```bash
.venv\Scripts\Activate.ps1
```

4. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

5. Download raw data
  - Go to [Kaggle Jigsaw Comemnt Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/overview)
  - Download data and extract all zip files until you get 4 .csv files (`train.csv`, `test.csv`, `test_labels.csv`, `sample_submission.csv`)
  - Add .csv files into `root/data/01_raw` folder. If it doesn't exist, create it manually

## 📊 Work in Progress

- [x] Project structure initialized
- [x] Virtual environment configured
- [x] Dependencies defined
- [x] Exploratory data analysis (EDA)
- [ ] Baseline model (TF-IDF + Logistic Regression)
- [ ] Transformer-based model (BERT)
- [ ] Evaluation and discussion
- [ ] Final report

## 🧠 Problem Summary

The challenge is to predict six independent toxicity labels per comment:

- toxic
- severe_toxic
-  obscene
- threat
- insult
- identity_hate

It is a multilabel classification problem, meaning that each comment can belong to none, one, or multiple categories simultaneously.

## 📌 Notes

- The virtual environment is not committed (see .gitignore).
- This repository is intended for evaluation purposes only.
- AI assistance tools (e.g., ChatGPT) were used for setup automation and documentation.

## 📬 Contact

For any questions about this solution, feel free to reach out via GitHub or email.