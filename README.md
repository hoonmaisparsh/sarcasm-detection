# Sarcasm Detection with Internet-Slang Understanding

Final-year B.Tech (CSE) project, Guru Nanak Institute of Technology (MAKAUT), Batch 2022–2026.

Sarcasm is hard for machines because the literal words often say the opposite of what is meant. On social media it is harder still, because people write in slang and abbreviations (`lol`, `smh`, `tbh`, `bruh`, `cap`...). This project asks one question:

> **Does teaching a sarcasm classifier about internet slang improve its performance?**

We build two pipelines on Reddit comments, one **without** slang understanding and one **with a custom internet-slang lexicon**, and compare them across classical ML models and an LSTM.

---

## Highlights

- Dataset: Reddit sarcasm corpus (`train-balanced-sarcasm`), using a 100,002-row working subset (`df_filtered.csv`; 57,692 non-sarcastic / 42,310 sarcastic).
- Baseline pipeline: text cleaning → TF-IDF (5,000 features) → Logistic Regression, Naive Bayes, Random Forest, SVM, XGBoost, Voting Classifier, LSTM.
- Lexicon-enhanced pipeline: adds an internet-slang lexicon, a `contains_slang` feature, and a **Lexicon-aware LSTM** (text sequence + slang feature as two inputs), with hyperparameters tuned using Keras Tuner (Hyperband).
- Analysis: slang prevalence in sarcastic vs. non-sarcastic comments, confusion matrices, ROC curves, and LIME explanations.

## Repository structure

```
.
├── notebooks/
│   ├── WITHOUT_SLANG_UNDERSTANDING.ipynb   # baseline pipeline
│   └── LEXICON_ENHANCED.ipynb              # slang-aware pipeline
├── data/
│   └── df_filtered.csv                     # 100,002-row working subset
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

> The full raw dataset (~255 MB) is **not** stored in this repo (GitHub's file limit is 100 MB). Download it from Kaggle: *Sarcasm on Reddit* (`train-balanced-sarcasm.csv`) and place it in `data/` if you want to re-create the subset.

## Method

1. **Preprocessing:** lowercase, strip punctuation and digits, tokenize (NLTK), remove stopwords.
2. **Features:** TF-IDF (max 5,000 features) for classical models; tokenized, padded sequences (vocab 5,000, length 100) for the LSTM.
3. **Slang lexicon:** a dictionary mapping common slang to meanings (e.g. `idk` → "I don't know", `smh` → "shaking my head"), used to flag and count slang in each comment.
4. **Models:** Logistic Regression, Multinomial Naive Bayes, Random Forest, Linear SVM, XGBoost, Voting Classifier, and a Lexicon-aware LSTM.
5. **Evaluation:** stratified 80/20 split (`random_state=42`, 20,000 test samples); accuracy, precision, recall and F1 for the sarcastic class.

## Results

**Lexicon-enhanced pipeline (test set, 20,000 comments):**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Multinomial NB | 0.6732 | 0.6863 | 0.4193 | 0.5205 |
| Logistic Regression | 0.6852 | 0.6682 | 0.5085 | 0.5775 |
| Random Forest | 0.6903 | 0.6880 | 0.4904 | 0.5727 |
| XGBoost | 0.6832 | 0.7152 | 0.4175 | 0.5272 |
| **Lexicon LSTM** | 0.6995 | 0.6868 | **0.5326** | **0.6000** |
| **Voting Classifier** | **0.7048** | **0.7361** | 0.4714 | 0.5747 |

**Baseline (no slang understanding):** Logistic Regression reached 0.6744 accuracy / 0.5546 F1, and the best classical accuracy was about 0.67. The plain LSTM did not learn the sarcastic class (recall 0.00), whereas the Lexicon-aware LSTM reached F1 = 0.60.

**Takeaway:** adding slang awareness gives a modest but consistent improvement, and the Lexicon LSTM has the best F1 and recall on the sarcastic class. Sarcasm detection from text alone remains hard, and accuracy stays around 70%.

## How to run

The notebooks were developed in **Google Colab**.

1. Open a notebook from `notebooks/` in Colab (or Jupyter).
2. Upload `data/df_filtered.csv` when prompted (Colab `files.upload()`), or change the path to read it from disk.
3. Run all cells. Note that the Keras Tuner search takes over an hour on a standard Colab runtime.

Local setup:

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

## Limitations and future work

- Only a 100k-comment subset is used, and the slang lexicon is small and hand-built.
- Comments are classified without their parent comment, even though context matters a lot for sarcasm.
- Future work: transformer models (BERT/RoBERTa), a larger lexicon or emoji handling, and using parent-comment context.

## Team

- Sparsh
- Ayush Kumar Gupta
- Randheer Thakur
- Rishi Kumar Sharma

**Supervisor:** Suman Bhattacharya

## Acknowledgements

Dataset: *Sarcasm on Reddit* (Khodak et al., "A Large Self-Annotated Corpus for Sarcasm", 2018).

## License

Released under the MIT License. See `LICENSE`.
