# Smart MCQ Solver Challenge

A Kaggle competition notebook (`DL-23f1000967-notebook-t22026`) that predicts the top-3 most likely correct answers for multiple-choice questions (5 options: A–E), evaluated using **MAP@3**.

## 📌 Problem Statement

Given a `prompt` (question) and five answer options (`A`, `B`, `C`, `D`, `E`), predict the top 3 most likely correct options in ranked order. Submissions are scored using **Mean Average Precision @ 3 (MAP@3)**.

## 📂 Data

| File | Description |
|---|---|
| `train.csv` | Training questions with `prompt`, options `A`–`E`, and the correct `answer` |
| `test.csv` | Test questions with `prompt` and options `A`–`E` (no answer) |
| `sample_submission.csv` | Submission format reference |

## 🔍 Approach

The notebook explores four progressively more sophisticated approaches:

### 1. TF-IDF Baseline
- Text cleaning (lowercasing, special character removal)
- TF-IDF vectorization (5000 features, 1–2 grams)
- Cosine similarity between prompt and each option to rank answers

### 2. Scratch Model (custom PyTorch)
- Hand-engineered features per option: word/char TF-IDF cosine similarity, SVD-reduced similarity, Jaccard similarity, length ratios, option centrality
- A custom feed-forward `OptionScorer` neural network (MLP with BatchNorm + Dropout) trained from scratch
- 5-fold Stratified Cross-Validation

### 3. DeBERTa-v3-base (fine-tuned)
- `microsoft/deberta-v3-base` fine-tuned using `AutoModelForMultipleChoice`
- 5-fold Stratified Cross-Validation, mixed precision training

### 4. RoBERTa-base (fine-tuned) — Final Model
- `roberta-base` fine-tuned using `AutoModelForMultipleChoice`
- 5-fold Stratified Cross-Validation, mixed precision training
- Produces the final `submission.csv`

## 📊 Evaluation Metric

**MAP@3 (Mean Average Precision @ 3)** — rewards correct predictions ranked higher among the top 3 guesses.

## 📈 Experiment Tracking

All experiments (loss curves, fold-wise MAP@3, EDA plots) are logged to **Weights & Biases (W&B)** under the project `DL-23f1000967-notebook-t22026`. The W&B API key is securely loaded via Kaggle Secrets (`WanDB_API_KEY`).

## 🛠️ Tech Stack

- **Data & ML:** pandas, numpy, scikit-learn
- **Deep Learning:** PyTorch, Hugging Face Transformers
- **Visualization:** matplotlib, seaborn
- **Experiment Tracking:** Weights & Biases (wandb)

## ▶️ How to Run

1. Open the notebook on Kaggle with GPU enabled.
2. Add your W&B API key under Kaggle Secrets as `WanDB_API_KEY`.
3. Run all cells sequentially — EDA → preprocessing → TF-IDF baseline → Scratch model → DeBERTa → RoBERTa.
4. The final cell generates `submission.csv`, ready for competition submission.

## 📝 Output

`submission.csv` with columns:
- `id` — question identifier
- `prediction` — top 3 predicted option labels, space-separated, ranked by confidence (e.g. `"A C B"`)
