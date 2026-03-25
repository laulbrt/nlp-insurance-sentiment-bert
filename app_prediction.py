import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
STOP = set(stopwords.words('english'))

st.set_page_config(page_title="Insurance Review Predictor", page_icon="⭐", layout="centered")

@st.cache_resource
def load_models():
    with open('model_tfidf_svc.pkl', 'rb') as f:
        star_model = pickle.load(f)
    with open('model_sentiment.pkl', 'rb') as f:
        sent_model = pickle.load(f)
    return star_model, sent_model

def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    return ' '.join(t for t in text.split() if t not in STOP and len(t) > 2)

@st.cache_resource
def load_zeroshot():
    from transformers import pipeline
    return pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

CATEGORIES = ['pricing', 'customer service', 'claims processing',
              'coverage', 'cancellation', 'enrollment', 'recommendation']

st.title("⭐ Insurance Review Predictor")
st.write("Enter an insurance review below and get predictions for star rating, sentiment, and topic.")

review_input = st.text_area("Your review", height=150,
    placeholder="e.g. The price is very high and the customer service never responds...")

col1, col2 = st.columns(2)
use_zeroshot = col2.checkbox("Also detect category (slower)", value=False)

if col1.button("Predict", type="primary") and review_input.strip():
    star_model, sent_model = load_models()
    cleaned = clean(review_input)

    star_pred = int(star_model.predict([cleaned])[0]) + 1
    stars_display = "⭐" * star_pred + "☆" * (5 - star_pred)

    sent_pred = sent_model.predict([cleaned])[0]
    sent_label = {0: "     :/ Negative", 1: "      -_- Neutral", 2: "     :) Positive"}[sent_pred]
    sent_color = {0: "red", 1: "orange", 2: "green"}[sent_pred]

    st.divider()
    st.subheader("Results")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Predicted rating", f"{star_pred} / 5")
        st.write(stars_display)
    with c2:
        st.markdown(f"**Sentiment** : :{sent_color}[{sent_label}]")

    st.subheader("SHAP – mots les plus influents")
    try:
        tfidf_vec   = star_model.named_steps['tfidf']
        clf         = star_model.named_steps['clf']
        feat_names  = tfidf_vec.get_feature_names_out()
        text_vec    = tfidf_vec.transform([cleaned])
        pred_class  = int(clf.predict(text_vec)[0])
        # SHAP values for linear model : coef * feature_value
        coef        = clf.coef_[pred_class]
        shap_vals   = text_vec.multiply(coef).toarray().flatten()
        nonzero_idx = text_vec.nonzero()[1]
        scores      = [(feat_names[i], float(shap_vals[i])) for i in nonzero_idx]
        scores      = sorted(scores, key=lambda x: abs(x[1]), reverse=True)[:10]

        words  = [s[0] for s in scores]
        values = [s[1] for s in scores]
        chart_df = pd.DataFrame({'word': words, 'SHAP value': values}).set_index('word')
        st.bar_chart(chart_df)
        st.caption("Valeurs SHAP : coef × TF-IDF (positif → pousse vers la classe prédite)")
    except Exception:
        st.write("(explanation not available)")

    if use_zeroshot:
        with st.spinner("Detecting category..."):
            zs = load_zeroshot()
            out = zs(review_input[:512], candidate_labels=CATEGORIES, multi_label=True)
            st.subheader("Detected topics")
            cat_df = pd.DataFrame({'category': out['labels'], 'score': out['scores']})
            cat_df = cat_df.sort_values('score', ascending=True).tail(5)
            st.bar_chart(cat_df.set_index('category'))

elif not review_input.strip() and st.session_state.get('predict_clicked'):
    st.warning("Please enter a review first.")
