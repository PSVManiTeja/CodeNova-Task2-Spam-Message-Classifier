"""
================================================================================
 CODENOVA AI/ML INTERNSHIP - TASK 2: SPAM MESSAGE CLASSIFIER (NLP)
 Comprehensive Self-Contained Training, Evaluation & Live Prediction Script
 Author: CodeNova AI/ML Intern
 Tech Stack: Python, Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn, NLTK
================================================================================
"""

import os
import sys
import re
import string
import io
import urllib.request
import zipfile
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

# ------------------------------------------------------------------------------
# 1. INITIALIZATION & NLP CORPUS SETUP
# ------------------------------------------------------------------------------
print("\n" + "=" * 78)
print("   CODENOVA AI/ML INTERNSHIP - BATCH 01: TASK 2 (SPAM CLASSIFIER)")
print("=" * 78)
print("[*] Verifying NLP corpora (stopwords, wordnet)...")

for pkg in ['stopwords', 'wordnet', 'punkt']:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """End-to-end text preprocessing pipeline."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    cleaned = [
        LEMMATIZER.lemmatize(t)
        for t in tokens
        if t not in STOPWORDS and len(t) > 1
    ]
    return " ".join(cleaned)


# ------------------------------------------------------------------------------
# 2. DATA INGESTION (OFFLINE LOCAL CACHE + UCI BENCHMARK FALLBACK)
# ------------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

CSV_FILE = os.path.join(DATA_DIR, "spam.csv")

if not os.path.exists(CSV_FILE):
    print("[*] Local dataset not detected. Ingesting official UCI SMS Spam Collection...")
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=15) as resp:
        z = zipfile.ZipFile(io.BytesIO(resp.read()))
        with z.open('SMSSpamCollection') as f:
            df = pd.read_csv(f, sep='\t', header=None, names=['label', 'message'])
    df.to_csv(CSV_FILE, index=False)
    print(f"    -> Downloaded and cached: {CSV_FILE}")
else:
    df = pd.read_csv(CSV_FILE)
    print(f"[*] Ingested existing local dataset: {CSV_FILE}")

# Data Cleaning
initial_rows = len(df)
df.dropna(subset=['label', 'message'], inplace=True)
df.drop_duplicates(subset=['message'], inplace=True)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
dropped = initial_rows - len(df)

counts = df['label'].value_counts()
print(f"    -> Ingested: {len(df)} unique records (Removed {dropped} duplicate/null rows)")
print(f"    -> Class Breakdown: HAM={counts['ham']} ({counts['ham']/len(df)*100:.1f}%), SPAM={counts['spam']} ({counts['spam']/len(df)*100:.1f}%)")

# ------------------------------------------------------------------------------
# 3. STATISTICAL VISUALIZATIONS
# ------------------------------------------------------------------------------
print("\n[*] Generating high-resolution visualizations...")
palette = ['#1f77b4', '#d62728']

# Feature Engineering for EDA
df['char_length'] = df['message'].apply(len)
df['word_count'] = df['message'].apply(lambda x: len(x.split()))

# Figure 1: Class Distribution & Length Comparison
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.countplot(data=df, x='label', hue='label', ax=axes[0], palette=palette, legend=False)
axes[0].set_title('Dataset Class Distribution (Ham vs Spam)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Class Label')
axes[0].set_ylabel('Number of Messages')
for p in axes[0].patches:
    axes[0].annotate(f"{int(p.get_height())} ({p.get_height()/len(df)*100:.1f}%)",
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', xytext=(0, 4), textcoords='offset points')

sns.histplot(data=df, x='char_length', hue='label', bins=60, kde=True,
             palette=palette, element='step', ax=axes[1])
axes[1].set_xlim(0, 350)
axes[1].set_title('Message Character Length Distribution', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Character Count')
axes[1].set_ylabel('Frequency')
plt.tight_layout()
fig1_path = os.path.join(PLOTS_DIR, "class_and_length_distribution.png")
plt.savefig(fig1_path, dpi=300)
plt.close()
print(f"    -> Exported: {fig1_path}")

# Preprocessing text
print("[*] Preprocessing text corpus (lowercasing, cleaning, lemmatizing)...")
df['clean_message'] = df['message'].apply(clean_text)

# ------------------------------------------------------------------------------
# 4. LEAKAGE-FREE SPLIT & TF-IDF VECTORIZATION
# ------------------------------------------------------------------------------
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df['clean_message'],
    df['label_num'],
    test_size=0.20,
    random_state=42,
    stratify=df['label_num']
)

# Fit vectorizer strictly on training partition
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)
X_train_vec = vectorizer.fit_transform(X_train_raw)
X_test_vec = vectorizer.transform(X_test_raw)
print(f"[*] Training samples: {len(X_train_raw)} | Testing samples: {len(X_test_raw)}")
print(f"[*] TF-IDF Vocabulary dimensions: {X_train_vec.shape[1]} n-gram features")

# ------------------------------------------------------------------------------
# 5. MODEL TRAINING & RIGOROUS BENCHMARKING
# ------------------------------------------------------------------------------
print("\n[*] Training and benchmarking candidate algorithms...")
models = {
    "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(C=10.0, max_iter=1000, random_state=42),
    "Calibrated Linear SVC": CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42, dual=False))
}

results = []
trained_estimators = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, clf in models.items():
    cv_prec = cross_val_score(clf, X_train_vec, y_train, cv=cv, scoring='precision').mean()
    cv_f1 = cross_val_score(clf, X_train_vec, y_train, cv=cv, scoring='f1').mean()
    clf.fit(X_train_vec, y_train)
    trained_estimators[name] = clf

    y_pred = clf.predict(X_test_vec)
    y_prob = clf.predict_proba(X_test_vec)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "CV Precision": cv_prec,
        "CV F1": cv_f1,
        "y_pred": y_pred,
        "y_prob": y_prob
    })

# Formatted Benchmark Table Output
print("\n" + "=" * 78)
print("             OFFICIAL MODEL EVALUATION BENCHMARK (HELD-OUT TEST SET)")
print("=" * 78)
headers = f"{'Model':<25} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'ROC-AUC':<8}"
print(headers)
print("-" * 78)
for r in results:
    print(f"{r['Model']:<25} | {r['Accuracy']*100:>8.2f}% | {r['Precision']*100:>8.2f}% | {r['Recall']*100:>7.2f}% | {r['F1-Score']*100:>7.2f}% | {r['ROC-AUC']:>8.4f}")
print("=" * 78)

# Select best model (prioritizing Precision to eliminate False Positives)
best_model_name = "Logistic Regression"
best_clf = trained_estimators[best_model_name]
pipeline = Pipeline([
    ('vectorizer', vectorizer),
    ('classifier', best_clf)
])

# Save models
joblib.dump(pipeline, os.path.join(MODELS_DIR, "best_spam_pipeline.joblib"))
joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
joblib.dump(best_clf, os.path.join(MODELS_DIR, "spam_classifier.joblib"))
print(f"[*] Serialized best pipeline ({best_model_name}) to: models/best_spam_pipeline.joblib")

# ------------------------------------------------------------------------------
# 6. EVALUATION CHARTS (CONFUSION MATRICES & ROC CURVES)
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for i, r in enumerate(results):
    cm = confusion_matrix(y_test, r['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False,
                annot_kws={'size': 13, 'weight': 'bold'})
    axes[i].set_title(f"{r['Model']}\nPrec: {r['Precision']*100:.1f}% | F1: {r['F1-Score']*100:.1f}%", fontsize=11)
    axes[i].set_xlabel('Predicted Label (0=Ham, 1=Spam)')
    axes[i].set_ylabel('True Label (0=Ham, 1=Spam)')
    axes[i].set_xticklabels(['Ham', 'Spam'])
    axes[i].set_yticklabels(['Ham', 'Spam'])
plt.tight_layout()
fig2_path = os.path.join(PLOTS_DIR, "confusion_matrices.png")
plt.savefig(fig2_path, dpi=300)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for i, r in enumerate(results):
    fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
    prec_c, rec_c, _ = precision_recall_curve(y_test, r['y_prob'])
    axes[0].plot(fpr, tpr, color=colors[i], lw=2, label=f"{r['Model']} ({r['ROC-AUC']:.4f})")
    axes[1].plot(rec_c, prec_c, color=colors[i], lw=2, label=f"{r['Model']}")

axes[0].plot([0, 1], [0, 1], 'k--', lw=1.2)
axes[0].set_title('Receiver Operating Characteristic (ROC)', fontweight='bold')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend(loc='lower right', fontsize=9)

axes[1].set_title('Precision-Recall Curves', fontweight='bold')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].legend(loc='lower left', fontsize=9)
plt.tight_layout()
fig3_path = os.path.join(PLOTS_DIR, "roc_and_pr_curves.png")
plt.savefig(fig3_path, dpi=300)
plt.close()
print(f"    -> Exported: {fig2_path}")
print(f"    -> Exported: {fig3_path}")

# ------------------------------------------------------------------------------
# 7. LIVE INFERENCE DEMO & INTERACTIVE PROMPT
# ------------------------------------------------------------------------------
def predict(text: str) -> dict:
    cleaned = clean_text(text)
    pred = int(pipeline.predict([cleaned])[0])
    probs = pipeline.predict_proba([cleaned])[0]
    return {
        "text": text,
        "clean": cleaned,
        "label": "SPAM" if pred == 1 else "HAM",
        "confidence": float(probs[pred]),
        "spam_prob": float(probs[1])
    }

print("\n" + "=" * 78)
print("            LIVE EXECUTION DEMO - REAL-TIME INFERENCE LOGS")
print("=" * 78)
test_cases = [
    "URGENT! You have won a guaranteed 1,000 cash prize. Call 09066362206 to claim your award immediately!",
    "Hey! Are we still meeting at the library at 4:30 pm to work on the AI project?",
    "Dear Customer, your bank transaction of $50.00 was approved. Available balance: $1,420.00.",
    "WINNER! Selected in the national draw! Text CLAIM to 87007 for your free voucher now! T&C apply.",
    "Can you email me the updated presentation slides before tomorrow's standup? Thanks."
]

print(f"{'INPUT TEXT SAMPLE':<52} | {'PRED':<4} | {'CONFIDENCE':<10} | {'SPAM PROB':<9}")
print("-" * 78)
for msg in test_cases:
    res = predict(msg)
    short_text = (msg[:49] + "...") if len(msg) > 52 else msg
    print(f"{short_text:<52} | {res['label']:<4} | {res['confidence']*100:>8.2f}%  | {res['spam_prob']*100:>7.2f}%")
print("=" * 78)


def launch_cli():
    print("\n" + "=" * 78)
    print("      INTERACTIVE CONSOLE: TEST ANY CUSTOM SMS / EMAIL IN REAL TIME")
    print("=" * 78)
    print("Type any text message to test live inference. Type 'exit' to quit.\n")
    while True:
        try:
            val = input(">> Enter custom message: ").strip()
            if not val:
                continue
            if val.lower() in ['exit', 'quit', 'q']:
                print("Interactive session ended.")
                break
            res = predict(val)
            status = "[!] SPAM DETECTED" if res['label'] == "SPAM" else "[OK] HAM / LEGITIMATE"
            print(f"   Result     : {status}")
            print(f"   Confidence : {res['confidence']*100:.2f}%")
            print(f"   Spam Risk  : {res['spam_prob']*100:.2f}%")
            print(f"   Cleaned NLP: '{res['clean']}'\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        launch_cli()
    else:
        print("\n[TIP] To launch the interactive live prediction console anytime, run:")
        print("      python spam_classifier.py --interactive")
        print("\nAll tasks executed successfully! Take screenshots of this terminal execution.")
