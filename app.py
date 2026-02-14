import streamlit as st
import joblib
import numpy as np


# Load model
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/tfidf.pkl")


# Page Config
st.set_page_config(
    page_title="AI Fake News Detector",
    layout="wide"
)


# Header
st.header("📰 AI Fake News Detection System")
st.write("An intelligent system to detect fake and real news using Machine Learning.")


# Sidebar
with st.sidebar:

    st.subheader("📌 Project Info")

    st.write("""
    This system uses:
    - TF-IDF Vectorization
    - Support Vector Machine
    - Machine Learning
    
    Developed for Internship Project.
    """)

    st.markdown("---")

    st.subheader("👤 User Details")

    name = st.text_input("Name")
    purpose = st.selectbox(
        "Purpose",
        ["Learning", "Research", "College Project", "Other"]
    )

    st.markdown("---")

    st.info("Enter news text in main area to analyze.")


# Tabs
tab1, tab2, tab3 = st.tabs(
    ["🔍 News Detection", "📊 Model Info", "ℹ️ About"]
)


# -------------------------
# TAB 1: Detection
# -------------------------

with tab1:

    st.subheader("Fake News Analyzer")

    news_text = st.text_area(
        "Paste News Article Here",
        height=200
    )

    if st.button("Analyze News"):

        if news_text.strip() == "":

            st.warning("Please enter some news text.")

        else:
            vec_text = vectorizer.transform([news_text])

            # Get prediction and score
            prediction = model.predict(vec_text)[0]
            score = model.decision_function(vec_text)[0]

            threshold = 0.3  # Confidence threshold

            confidence = round(abs(score) * 100, 2)

            # Check uncertainty
            if abs(score) < threshold:
                st.warning("⚠️ Prediction Uncertain")

            else:
                # Use prediction for class
                if prediction == 1:
                    st.error("❌ This News is FAKE")
                else:
                    st.success("✅ This News is REAL")

            st.write(f"Confidence Score: {confidence}%")




# -------------------------
# TAB 2: Model Info
# -------------------------

with tab2:

    st.subheader("Model Details")

    st.write("""
    ### Algorithm
    - Support Vector Machine (LinearSVC)

    ### Feature Extraction
    - TF-IDF Vectorizer
    - Unigrams and Bigrams

    ### Dataset
    - WELFake Dataset 

    ### Evaluation
    - Accuracy: ~96.8%
    - Precision, Recall, F1-score used
    """)


# -------------------------
# TAB 3: About
# -------------------------

with tab3:

    st.subheader("About This Project")

    st.write("""
    This project was developed as part of an AI & ML internship.

    Objectives:
    - Detect fake news
    - Reduce misinformation
    - Apply ML in real-world scenario

    Domain Perspective & Limitations:
    - The model is trained mainly on political and online news data.
    - It performs best on articles related to government, elections, and public policy.
    - Performance may decrease on out-of-domain topics such as sports, entertainment, or science.
    - Predictions are based on learned text patterns, not factual verification.

    Note:
    - An uncertainty mechanism is included to handle low-confidence predictions.

    Future Scope:
    - Deep Learning models (BERT, RoBERTa)
    - Multi-source data integration
    - Real-time fact-checking APIs
    """)
