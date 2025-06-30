# 📝 Yelp Sentiment Analysis + Explainability

This project performs sentiment analysis on Yelp reviews using machine learning and deep learning models. It also includes model explainability techniques like SHAP and Integrated Gradients to interpret model predictions.

---

## 📚 Dataset

A subset of the **Yelp Review Dataset** is used for binary sentiment classification:

- **0** = Negative (1–2 star reviews)  
- **1** = Positive (4–5 star reviews)  
- 3-star reviews are excluded to avoid ambiguity.

---

## 🚀 Project Highlights

### 🔄 Preprocessing
- Text cleaning, tokenization, stopword removal
- Vectorization with:
  - **TF-IDF**
  - **Word2Vec**

### 🤖 Models
- Logistic Regression
- LSTM
- DistilBERT (Transformer-based)

### 📊 Evaluation
- Accuracy
- F1 Score
- Confusion Matrix

### 🔍 Model Explainability
- **SHAP (SHapley Additive exPlanations)**: Feature attribution for tree- and linear-based models  
- **Integrated Gradients**: Used with deep learning models (e.g., LSTM, DistilBERT) to trace input importance

---

## 📁 Files

- `sentimentanalysis_YELP1.ipynb` – Full notebook (data processing → model training → explainability)
- `README.md` – Project overview

---

## 🛠 Requirements

Install dependencies via pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk gensim torch transformers shap captum
