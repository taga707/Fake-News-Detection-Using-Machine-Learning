"""
Train and save the Fake News Detection models — on the CLEANED dataset.

Unlike the original script, this version does not merge separate
Fake.csv / True.csv files. It trains directly on the already-cleaned,
de-duplicated, non-truncated `dataset.csv` produced by the data-cleaning
pass (see dataset_improved.csv for the full cleaning log / flagged rows).

Cleaning applied upstream, before this script runs:
  - exact duplicate rows removed
  - duplicate title+text articles removed
  - empty / near-empty articles (<20 chars) removed
  - rows truncated at exactly 680 characters (a data artifact from the
    original source file) removed entirely, since a model trained on
    cut-off sentences learns cut-off patterns
  - whitespace stripped from all text fields
  - dates standardized to ISO format

Run this to regenerate:
  fake_news_models.pkl   (dict of the 3 trained models)
  tfidf_vectorizer.pkl
  model_metadata.pkl
"""

import re
import string
import pickle
import warnings

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings('ignore')

print("=" * 60)
print("FAKE NEWS DETECTION - MODEL TRAINING (cleaned dataset)")
print("=" * 60)

# 1. Load the cleaned, merged dataset
print("\n1. Loading cleaned dataset...")
df = pd.read_csv('dataset.csv')
print(f"   Shape: {df.shape}")
print(f"   Class balance: {df['class'].value_counts().to_dict()}")

df = df[['text', 'class']].dropna()

# 2. Shuffle
print("\n2. Shuffling...")
df = df.sample(frac=1, random_state=42)
df.reset_index(drop=True, inplace=True)

# 3. Text cleaning (same as notebook's wordopt)
print("\n3. Cleaning text...")


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


df["text"] = df["text"].apply(wordopt)
print("   Done.")

# 4. Features / target
x = df["text"]
y = df["class"]

# 5. Train/test split (75/25, as in the notebook)
print("\n4. Splitting dataset (75/25)...")
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42, stratify=y
)
print(f"   Train: {x_train.shape[0]}   Test: {x_test.shape[0]}")

# 6. TF-IDF vectorization (vocabulary capped to keep the vectorizer + models small)
print("\n5. Vectorizing text (TF-IDF, max_features=3000)...")
vectorization = TfidfVectorizer(max_features=3000, min_df=2, max_df=0.9)
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)
print(f"   Vocabulary size: {len(vectorization.vocabulary_):,}")

# 7. Train the three models
results = {}
models = {}

print("\n6. Training Logistic Regression...")
LR = LogisticRegression(max_iter=1000)
LR.fit(xv_train, y_train)
models['Logistic Regression'] = LR

print("\n7. Training Decision Tree (max_depth=20, min_samples_leaf=10)...")
DT = DecisionTreeClassifier(max_depth=20, min_samples_leaf=10, random_state=0)
DT.fit(xv_train, y_train)
models['Decision Tree'] = DT

print("\n8. Training Random Forest (n_estimators=40, max_depth=15, min_samples_leaf=10)...")
RFC = RandomForestClassifier(n_estimators=40, max_depth=15, min_samples_leaf=10, random_state=0)
RFC.fit(xv_train, y_train)
models['Random Forest'] = RFC

# 8. Evaluate all three
print("\n9. Evaluating models...")
for name, clf in models.items():
    pred = clf.predict(xv_test)
    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    report = classification_report(y_test, pred, output_dict=True)
    results[name] = {
        'accuracy': acc,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
    }
    print(f"   {name}: accuracy = {acc:.4f}")

# 9. Save artifacts
print("\n10. Saving artifacts...")
with open('fake_news_models.pkl', 'wb') as f:
    pickle.dump(models, f)
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorization, f)

metadata = {
    'results': results,
    'labels': {0: 'Fake News', 1: 'Not A Fake News'},
    'n_train': int(x_train.shape[0]),
    'n_test': int(x_test.shape[0]),
    'n_features': len(vectorization.vocabulary_),
    'class_counts': df['class'].value_counts().to_dict(),
}
with open('model_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("    Saved fake_news_models.pkl, tfidf_vectorizer.pkl, model_metadata.pkl")
print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE — run: streamlit run app.py")
print("=" * 60)
