# 📝 Sentiment-analysis-using-YELP

This project analyzes Yelp reviews to determine sentiment (positive or negative) using traditional and deep learning methods.

## 📚 Dataset
The notebook uses a processed subset of the **Yelp Review Dataset**, labeled for binary sentiment classification.

- Labels:
  - `0`: Negative (1–2 star reviews)
  - `1`: Positive (4–5 star reviews)
  - 3-star reviews are excluded.

## 🚀 Models & Methods
- **Text Preprocessing**
  - Lowercasing, punctuation removal, stopword removal
  - Tokenization & Vectorization (TF-IDF, Word2Vec)
- **Classification Models**
  - Logistic Regression
  - LSTM
  - DistilBERT
- **Evaluation**
  - Accuracy, F1 Score, Confusion Matrix

## 📁 Files
- `sentimentanalysis_YELP1.ipynb`: Main notebook with end-to-end pipeline

## 🛠 Requirements

Install dependencies with:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk gensim torch transformers
