# 📰 Fake News Detection System (Machine Learning)

<img width="1913" height="962" alt="Screenshot 2026-02-15 010711" src="https://github.com/user-attachments/assets/ab34fe25-0601-4491-944b-6d7f3ad7bd34" />


A web-based application that detects fake and real news articles using Natural Language Processing and Machine Learning techniques.

This project was developed as part of an AI & ML Internship.

---

## 🚀 Demo

- Deployment: Local Streamlit Application
- URL: http://localhost:8501 (when running locally)

---

## 📌 Problem Statement

With the rapid growth of social media, misinformation spreads quickly. Users often find it difficult to verify the authenticity of news articles. Manual fact-checking is time-consuming and unreliable.

This project aims to provide an automated solution for detecting fake news using Machine Learning.

---

## 💡 Solution Overview

The system analyzes news text and classifies it as Fake or Real using:

- TF-IDF for feature extraction
- Support Vector Machine (SVM) for classification
- Confidence-based uncertainty handling
- Streamlit for deployment

---

## 📊 Dataset

- Name: WELFake Dataset
- Source: Zenodo
- Type: Labeled news dataset
- Labels:
  - 0 → Real
  - 1 → Fake

---

## ⚙️ Technologies Used

| Category        | Tools / Libraries       |
|-----------------|--------------------------|
| Programming     | Python                   |
| Data Processing | Pandas, NumPy            |
| ML / NLP        | Scikit-learn, NLTK       |
| Deployment      | Streamlit                |
| IDE             | VS Code, Jupyter Notebook|

---

## 🧠 System Architecture

```
User Input
     ↓
Text Preprocessing
     ↓
TF-IDF Vectorizer
     ↓
SVM Classifier
     ↓
Prediction Output
```

---

## 🔄 Project Workflow

1. Data Collection
2. Data Cleaning & Preprocessing
3. Feature Engineering (TF-IDF)
4. Model Training (SVM)
5. Model Evaluation
6. Model Saving (Joblib)
7. Web Deployment (Streamlit)

---

## 🗂️ Project Structure

```
FakeNewsProject/
│
├── app.py
├── train.ipynb
├── requirements.txt
├── README.md
│
├── model/
│   ├── fake_news_model.pkl
│   └── tfidf.pkl
│
└── data/
    └── WELFake_Dataset.csv
```

---

## 📈 Results

- Best Model: Support Vector Machine (LinearSVC)
- Accuracy: ~96.8%
- Strong performance on benchmark dataset
- Includes uncertainty handling for low-confidence cases

---

## ⚠️ Limitations

- Trained mainly on political and online news data
- Performance decreases on out-of-domain topics
- No real-time fact verification
- Dataset bias affects predictions

---

## 🔮 Future Improvements

- Integration of transformer-based models (BERT, RoBERTa)
- Real-time fact-checking APIs
- Multi-language support
- Cloud deployment

---

## ▶️ How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/YourUsername/fake-news-detector-ml.git
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run app.py
```

---

## 👨‍💻 Author

**Name:** Mann Raval  
**Program:** AI & ML Internship  

---

## 📚 References

- WELFake Dataset (Zenodo)
- Scikit-learn Documentation
- Streamlit Documentation
- Python Official Documentation
