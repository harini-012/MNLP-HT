
import re
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Public copy of the SMS Spam Collection dataset
DATASET_URL = (
    "https://raw.githubusercontent.com/"
    "justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
)

print("Loading SMS Spam Collection dataset...")

df = pd.read_csv(
    DATASET_URL,
    sep="\t",
    header=None,
    names=["label", "message"],
    usecols=[0, 1],
    on_bad_lines="skip",
)

df = df.dropna(subset=["label", "message"])
df["label"] = df["label"].str.lower().str.strip()
df = df[df["label"].isin(["ham", "spam"])]

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["message"] = df["message"].apply(clean_text)

# Convert labels: ham = 0, spam = 1
df["label"] = df["label"].map({"ham": 0, "spam": 1})

X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Convert messages into TF-IDF features
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=10000,
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the classifier
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_tfidf, y_train)

# Evaluate
predictions = model.predict(X_test_tfidf)

print("\nModel Evaluation")
print("----------------")
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["ham", "spam"],
        zero_division=0,
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Save model and vectorizer
joblib.dump(model, "spam_model.joblib")
joblib.dump(vectorizer, "tfidf_vectorizer.joblib")

print("\nModel saved as spam_model.joblib")
print("Vectorizer saved as tfidf_vectorizer.joblib")
