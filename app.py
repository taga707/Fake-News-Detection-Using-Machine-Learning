"""
Fake News Detection using Machine Learning — Enhanced Edition
Streamlit web application: dataset explorer, rich EDA, live prediction with
explainability, batch prediction, model comparison & live evaluation.

Pipeline mirrors the notebook: TF-IDF + Logistic Regression / Decision Tree /
Random Forest.
"""

import re
import string
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# ---------------------------------------------------------------------------
# Page config & global styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fake News Detection AI",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main > div { padding-top: 1rem; }
    .stMetric {
        background: rgba(120, 120, 120, 0.08);
        border-radius: 12px;
        padding: 12px 8px;
        border: 1px solid rgba(120,120,120,0.15);
    }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    .hero {
        padding: 1.6rem 2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        color: #f9fafb;
        margin-bottom: 1.2rem;
    }
    .hero h1 { margin: 0 0 0.3rem 0; font-size: 2rem; }
    .hero p { margin: 0; opacity: 0.85; }
    .pill {
        display: inline-block; padding: 3px 12px; border-radius: 999px;
        font-size: 0.8rem; font-weight: 600; margin-right: 6px;
    }
    .pill-fake { background: #fee2e2; color: #991b1b; }
    .pill-real { background: #dcfce7; color: #166534; }
    .footer-note { opacity: 0.6; font-size: 0.8rem; margin-top: 2rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because been
before being below between both but by can't cannot could couldn't did didn't do does
doesn't doing don't down during each few for from further had hadn't has hasn't have
haven't having he he'd he'll he's her here here's hers herself him himself his how
how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most
mustn't my myself no nor not of off on once only or other ought our ours ourselves out
over own same shan't she she'd she'll she's should shouldn't so some such than that
that's the their theirs them themselves then there there's these they they'd they'll
they're they've this those through to too under until up very was wasn't we we'd we'll
we're we've were weren't what what's when when's where where's which while who who's
whom why why's with won't would wouldn't you you'd you'll you're you've your yours
yourself yourselves said say says one also new will us reuters
""".split())


def wordopt(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text


@st.cache_resource
def load_model_and_metadata():
    with open('fake_news_models.pkl', 'rb') as f:
        models = pickle.load(f)   # dict: name -> fitted classifier
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    return models, vectorizer, metadata


@st.cache_data
def load_default_dataset():
    return pd.read_csv('dataset.csv')


@st.cache_data
def top_words_by_label(df, label_col='class', text_col='text', n=20):
    """Return top-N most frequent non-stopword tokens for each class."""
    out = {}
    sample = df.sample(min(6000, len(df)), random_state=42)
    for cls_val, group in sample.groupby(label_col):
        counter = Counter()
        for t in group[text_col].astype(str).head(3000):
            cleaned = wordopt(t)
            tokens = [w for w in cleaned.split() if len(w) > 2 and w not in STOPWORDS]
            counter.update(tokens)
        out[cls_val] = counter.most_common(n)
    return out


models, vectorizer, metadata = load_model_and_metadata()
LABELS = metadata['labels']  # {0: 'Fake News', 1: 'Not A Fake News'}

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("📰 Fake News AI")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📊 Dataset Explorer", "📈 Data Analysis",
     "🔮 Predictions", "📂 Batch Prediction", "ℹ️ Model Info"],
)

st.sidebar.markdown("---")
st.sidebar.caption("TF-IDF + Logistic Regression / Decision Tree / Random Forest")

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: text, model, prediction, confidence

with st.sidebar.expander("🕘 Recent predictions", expanded=False):
    if st.session_state.history:
        for h in reversed(st.session_state.history[-8:]):
            tag = "🔴" if h["prediction"] == LABELS[0] else "🟢"
            st.caption(f"{tag} {h['model']} · {h['text_preview']}")
        if st.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("No predictions yet.")

# ---------------------------------------------------------------------------
# HOME
# ---------------------------------------------------------------------------
if page == "🏠 Home":
    st.markdown(
        """
        <div class="hero">
            <h1>📰 Fake News Detection using Machine Learning</h1>
            <p>Classify news articles as real or fake using three trained ML models,
            explore the training data, and see how each model reasons about a prediction.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    best_model = max(metadata['results'], key=lambda k: metadata['results'][k]['accuracy'])
    best_acc = metadata['results'][best_model]['accuracy']

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Model", best_model)
    col2.metric("Best Accuracy", f"{best_acc*100:.2f}%")
    col3.metric("Training Records", f"{metadata['n_train']:,}")
    col4.metric("TF-IDF Vocabulary", f"{metadata['n_features']:,}")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("What this app does")
        st.markdown("""
        - 📊 **Dataset Explorer** — browse, search, and filter the merged True/Fake news dataset
        - 📈 **Data Analysis** — class balance, article length, subject & word-frequency breakdown
        - 🔮 **Predictions** — paste an article, get a confidence score and word-level explanation
        - 📂 **Batch Prediction** — upload a CSV of articles and classify them all at once
        - ℹ️ **Model Info** — accuracy, confusion matrix, ROC curve, classification report
        """)
    with c2:
        st.subheader("How it works")
        st.markdown("""
        ```
        Raw Article Text
              ↓
        Text Cleaning (lowercase, strip URLs/HTML/punctuation/digits)
              ↓
        TF-IDF Vectorization
              ↓
        Logistic Regression / Decision Tree / Random Forest
              ↓
        Real / Fake Prediction + Confidence
        ```
        """)

    st.subheader("Model Comparison")
    comp_df = pd.DataFrame({
        'Model': list(metadata['results'].keys()),
        'Accuracy': [v['accuracy'] for v in metadata['results'].values()],
    }).sort_values('Accuracy', ascending=False)
    fig, ax = plt.subplots(figsize=(7, 2.6))
    bars = ax.barh(comp_df['Model'], comp_df['Accuracy'] * 100,
                    color=["#2563eb" if m == best_model else "#93c5fd" for m in comp_df['Model']])
    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(0, 100)
    for bar, v in zip(bars, comp_df['Accuracy'] * 100):
        ax.text(v + 0.5, bar.get_y() + bar.get_height() / 2, f"{v:.2f}%", va='center', fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    st.pyplot(fig, use_container_width=True)

    st.info("👈 Use the sidebar to explore the dataset, view analysis, or try a live prediction.")

# ---------------------------------------------------------------------------
# DATASET EXPLORER
# ---------------------------------------------------------------------------
elif page == "📊 Dataset Explorer":
    st.title("📊 Dataset Explorer")

    uploaded = st.file_uploader("Upload your own CSV (optional)", type="csv")
    df = pd.read_csv(uploaded) if uploaded is not None else load_default_dataset()

    st.success(f"Loaded dataset with **{df.shape[0]:,}** rows and **{df.shape[1]}** columns.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{df.shape[0]:,}")
    c2.metric("Total Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))
    c4.metric("Duplicate Rows", int(df.duplicated().sum()))

    st.markdown("### 🔍 Search & Filter")
    fc1, fc2, fc3 = st.columns([2, 1, 1])
    with fc1:
        query = st.text_input("Search in title/text", "")
    with fc2:
        label_filter = st.multiselect(
            "Label", options=sorted(df['label'].unique()) if 'label' in df.columns else [],
        )
    with fc3:
        subject_filter = st.multiselect(
            "Subject", options=sorted(df['subject'].unique()) if 'subject' in df.columns else [],
        )

    filtered = df.copy()
    if query:
        mask = pd.Series(False, index=filtered.index)
        for col in ['title', 'text']:
            if col in filtered.columns:
                mask |= filtered[col].astype(str).str.contains(query, case=False, na=False)
        filtered = filtered[mask]
    if label_filter and 'label' in filtered.columns:
        filtered = filtered[filtered['label'].isin(label_filter)]
    if subject_filter and 'subject' in filtered.columns:
        filtered = filtered[filtered['subject'].isin(subject_filter)]

    st.caption(f"Showing **{len(filtered):,}** of {len(df):,} rows after filters.")

    st.subheader("Preview")
    n_rows = st.slider("Rows to display", 5, 200, 10)
    st.dataframe(filtered.head(n_rows), use_container_width=True)

    st.download_button(
        "⬇️ Download filtered data as CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_news.csv",
        mime="text/csv",
    )

    if 'label' in df.columns:
        st.subheader("Label Distribution (filtered)")
        st.bar_chart(filtered['label'].value_counts())

# ---------------------------------------------------------------------------
# DATA ANALYSIS (EDA)
# ---------------------------------------------------------------------------
elif page == "📈 Data Analysis":
    st.title("📈 Exploratory Data Analysis")
    df = load_default_dataset()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Label Distribution", "By Subject", "Article Length", "Top Words", "Timeline"]
    )

    with tab1:
        st.subheader("Real vs Fake Distribution")
        fig, ax = plt.subplots(figsize=(7, 5))
        counts = df['label'].value_counts()
        colors = ['#FF6B6B', '#4ECDC4']
        bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='black', linewidth=1.5)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h, f'{int(h)}', ha='center', va='bottom', fontweight='bold')
        ax.set_xlabel("Label")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Fake vs Real News")
        ax.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig, use_container_width=True)

    with tab2:
        st.subheader("Article Count by Subject")
        st.bar_chart(df['subject'].value_counts())

        st.subheader("Subject vs Label Breakdown")
        cross = pd.crosstab(df['subject'], df['label'])
        fig, ax = plt.subplots(figsize=(9, 4))
        cross.plot(kind='bar', stacked=True, ax=ax, color=['#FF6B6B', '#4ECDC4'])
        ax.set_ylabel("Count")
        ax.tick_params(axis='x', rotation=45)
        ax.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig, use_container_width=True)

    with tab3:
        st.subheader("Article Length Distribution (words)")
        sample = df.sample(min(5000, len(df)), random_state=42).copy()
        sample['word_count'] = sample['text'].astype(str).str.split().apply(len)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data=sample, x='word_count', hue='label', bins=40, kde=True, ax=ax)
        ax.set_xlim(0, sample['word_count'].quantile(0.98))
        ax.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig, use_container_width=True)

        st.subheader("Length Summary Stats")
        stats = sample.groupby('label')['word_count'].describe()[['mean', '50%', 'min', 'max']]
        stats.columns = ['Mean', 'Median', 'Min', 'Max']
        st.dataframe(stats.round(1), use_container_width=True)

    with tab4:
        st.subheader("Most Frequent Words by Class")
        st.caption("Stopwords removed · computed on a random sample for speed")
        word_data = top_words_by_label(df)
        wc1, wc2 = st.columns(2)
        for cls_val, col in zip(sorted(word_data.keys()), [wc1, wc2]):
            with col:
                st.markdown(f"**{LABELS.get(cls_val, cls_val)}**")
                pairs = word_data[cls_val]
                words = [p[0] for p in pairs][::-1]
                counts = [p[1] for p in pairs][::-1]
                fig, ax = plt.subplots(figsize=(5, 6))
                color = '#FF6B6B' if cls_val == 0 else '#4ECDC4'
                ax.barh(words, counts, color=color)
                ax.spines[['top', 'right']].set_visible(False)
                st.pyplot(fig, use_container_width=True)

    with tab5:
        st.subheader("Articles Over Time")
        dated = df.copy()
        dated['parsed_date'] = pd.to_datetime(dated['date'], errors='coerce', format='mixed')
        dated = dated.dropna(subset=['parsed_date'])
        if dated.empty:
            st.warning("No parseable dates found in this dataset.")
        else:
            monthly = dated.set_index('parsed_date').groupby([pd.Grouper(freq='ME'), 'label']).size().unstack(fill_value=0)
            fig, ax = plt.subplots(figsize=(9, 4))
            monthly.plot(ax=ax, color=['#FF6B6B', '#4ECDC4'], linewidth=2)
            ax.set_ylabel("Article count")
            ax.set_xlabel("Month")
            ax.spines[['top', 'right']].set_visible(False)
            st.pyplot(fig, use_container_width=True)
            st.caption(f"Parsed {len(dated):,} of {len(df):,} rows ({len(dated)/len(df)*100:.0f}%).")

# ---------------------------------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------------------------------
elif page == "🔮 Predictions":
    st.title("🔮 Real-Time Fake News Prediction")
    st.markdown("Paste a news article's text below, choose a model, and click **Predict**.")

    text = st.text_area(
        "Article text",
        "WASHINGTON (Reuters) - The U.S. Senate voted on Tuesday to advance a bipartisan "
        "infrastructure bill after months of negotiations between lawmakers.",
        height=200,
    )
    col_a, col_b = st.columns([2, 1])
    with col_a:
        model_choice = st.selectbox("Model", list(models.keys()))
    with col_b:
        st.metric("Word count", len(text.split()))

    if st.button("🚀 Predict", type="primary", use_container_width=True):
        cleaned = wordopt(text)
        vec = vectorizer.transform([cleaned])
        clf = models[model_choice]
        pred = clf.predict(vec)[0]
        label = LABELS[pred]

        proba = clf.predict_proba(vec)[0] if hasattr(clf, "predict_proba") else None
        confidence = float(proba[pred]) if proba is not None else None

        st.session_state.history.append({
            "text_preview": (text[:40] + "…") if len(text) > 40 else text,
            "model": model_choice,
            "prediction": label,
            "confidence": confidence,
        })

        st.markdown("---")
        pill_class = "pill-fake" if pred == 0 else "pill-real"
        icon = "⚠️" if pred == 0 else "✅"
        conf_txt = f" · confidence {confidence*100:.1f}%" if confidence is not None else ""
        st.markdown(
            f'<span class="pill {pill_class}">{icon} {label}</span> '
            f'<span style="opacity:0.7">via {model_choice}{conf_txt}</span>',
            unsafe_allow_html=True,
        )

        if proba is not None:
            prob_df = pd.DataFrame({
                'Label': [LABELS[0], LABELS[1]],
                'Probability': proba,
            }).set_index('Label')
            st.bar_chart(prob_df)

        # --- Explainability for linear model ---
        if model_choice == "Logistic Regression" and hasattr(clf, "coef_"):
            st.subheader("🔬 Why this prediction? (top contributing words)")
            feature_names = np.array(vectorizer.get_feature_names_out())
            coefs = clf.coef_[0]
            present_idx = vec.nonzero()[1]
            if len(present_idx) > 0:
                contrib = coefs[present_idx] * vec.toarray()[0][present_idx]
                order = np.argsort(contrib)
                top_neg = order[:8]   # pushes toward class 0 (Fake)
                top_pos = order[-8:]  # pushes toward class 1 (Not Fake)

                exp_df = pd.DataFrame({
                    "word": np.concatenate([feature_names[present_idx][top_neg],
                                             feature_names[present_idx][top_pos]]),
                    "influence": np.concatenate([contrib[top_neg], contrib[top_pos]]),
                }).sort_values("influence")

                fig, ax = plt.subplots(figsize=(7, 4))
                colors = ['#FF6B6B' if v < 0 else '#4ECDC4' for v in exp_df['influence']]
                ax.barh(exp_df['word'], exp_df['influence'], color=colors)
                ax.axvline(0, color='black', linewidth=0.8)
                ax.set_xlabel(f"← pushes toward '{LABELS[0]}'   |   pushes toward '{LABELS[1]}' →")
                ax.spines[['top', 'right']].set_visible(False)
                st.pyplot(fig, use_container_width=True)
            else:
                st.caption("None of the article's words were found in the model's vocabulary.")

        st.subheader("Compare across all models")
        rows = []
        for name, m in models.items():
            p = m.predict(vec)[0]
            row = {'Model': name, 'Prediction': LABELS[p]}
            if hasattr(m, "predict_proba"):
                row['Confidence'] = f"{m.predict_proba(vec)[0][p]*100:.1f}%"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# BATCH PREDICTION
# ---------------------------------------------------------------------------
elif page == "📂 Batch Prediction":
    st.title("📂 Batch Prediction")
    st.markdown("Upload a CSV with a `text` column to classify many articles at once.")

    model_choice = st.selectbox("Model to use", list(models.keys()), key="batch_model")
    batch_file = st.file_uploader("Upload CSV", type="csv", key="batch_upload")

    if batch_file is not None:
        batch_df = pd.read_csv(batch_file)
        if 'text' not in batch_df.columns:
            st.error("The uploaded CSV must contain a `text` column.")
        else:
            st.success(f"Loaded {len(batch_df):,} rows.")
            if st.button("🚀 Run batch prediction", type="primary"):
                with st.spinner("Classifying articles..."):
                    cleaned = batch_df['text'].astype(str).apply(wordopt)
                    vecs = vectorizer.transform(cleaned)
                    clf = models[model_choice]
                    preds = clf.predict(vecs)
                    batch_df['prediction'] = [LABELS[p] for p in preds]
                    if hasattr(clf, "predict_proba"):
                        probs = clf.predict_proba(vecs)
                        batch_df['confidence'] = [f"{probs[i][p]*100:.1f}%" for i, p in enumerate(preds)]

                st.dataframe(batch_df, use_container_width=True)

                c1, c2 = st.columns(2)
                c1.metric("Predicted Fake", int((preds == 0).sum()))
                c2.metric("Predicted Real", int((preds == 1).sum()))

                st.bar_chart(batch_df['prediction'].value_counts())

                st.download_button(
                    "⬇️ Download results as CSV",
                    data=batch_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"batch_predictions_{model_choice.replace(' ', '_').lower()}.csv",
                    mime="text/csv",
                )

# ---------------------------------------------------------------------------
# MODEL INFO
# ---------------------------------------------------------------------------
elif page == "ℹ️ Model Info":
    st.title("ℹ️ Model Information")

    model_choice = st.selectbox("Choose a model to inspect", list(metadata['results'].keys()))
    res = metadata['results'][model_choice]

    c1, c2, c3 = st.columns(3)
    c1.metric("Model", model_choice)
    c2.metric("Accuracy", f"{res['accuracy']*100:.2f}%")
    c3.metric("Test Records", f"{metadata['n_test']:,}")

    ic1, ic2 = st.columns(2)
    with ic1:
        st.subheader("Confusion Matrix")
        cm = np.array(res['confusion_matrix'])
        label_names = [LABELS[0], LABELS[1]]
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=label_names, yticklabels=label_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig, use_container_width=True)

    with ic2:
        st.subheader("Classification Report")
        report_df = pd.DataFrame(res['classification_report']).T
        report_df = report_df.rename(index={'0': LABELS[0], '1': LABELS[1]})
        st.dataframe(report_df.round(3), use_container_width=True)

    st.markdown("---")
    st.subheader("Precision / Recall / F1 — All Models")
    metric_rows = []
    for name, r in metadata['results'].items():
        rep = r['classification_report']['weighted avg']
        metric_rows.append({
            'Model': name,
            'Precision': rep['precision'],
            'Recall': rep['recall'],
            'F1-score': rep['f1-score'],
            'Accuracy': r['accuracy'],
        })
    metric_df = pd.DataFrame(metric_rows).set_index('Model')
    fig, ax = plt.subplots(figsize=(9, 4))
    metric_df.plot(kind='bar', ax=ax, color=['#60a5fa', '#34d399', '#fbbf24', '#f87171'])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    ax.tick_params(axis='x', rotation=0)
    ax.spines[['top', 'right']].set_visible(False)
    st.pyplot(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📉 Live ROC Curve (evaluated on current dataset)")
    st.caption(
        "Computed on the fly against `dataset.csv` for models that support probability "
        "output. This is a live sanity check and may differ slightly from the stored "
        "training-time metrics above if the dataset has changed."
    )
    df_eval = load_default_dataset()
    if st.button("Run live ROC evaluation"):
        with st.spinner("Scoring dataset with all models..."):
            sample_eval = df_eval.sample(min(4000, len(df_eval)), random_state=42)
            y_true = sample_eval['class'].values
            cleaned_eval = sample_eval['text'].astype(str).apply(wordopt)
            X_eval = vectorizer.transform(cleaned_eval)

            fig, ax = plt.subplots(figsize=(6, 5))
            for name, m in models.items():
                if hasattr(m, "predict_proba"):
                    scores = m.predict_proba(X_eval)[:, 1]
                    fpr, tpr, _ = roc_curve(y_true, scores)
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", linewidth=2)
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend(loc='lower right')
            ax.spines[['top', 'right']].set_visible(False)
            st.pyplot(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Overall Accuracy Comparison")
    comp_df = pd.DataFrame({
        'Model': list(metadata['results'].keys()),
        'Accuracy': [v['accuracy'] for v in metadata['results'].values()],
    }).sort_values('Accuracy', ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(comp_df['Model'], comp_df['Accuracy'] * 100, color="steelblue")
    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(0, 100)
    for i, v in enumerate(comp_df['Accuracy'] * 100):
        ax.text(v + 0.5, i, f"{v:.2f}%", va='center')
    ax.spines[['top', 'right']].set_visible(False)
    st.pyplot(fig, use_container_width=True)

    st.caption(f"Trained on {metadata['n_train']:,} records · Tested on {metadata['n_test']:,} records")

st.markdown(
    '<p class="footer-note">Fake News Detection AI · TF-IDF + Logistic Regression / '
    'Decision Tree / Random Forest</p>',
    unsafe_allow_html=True,
)
