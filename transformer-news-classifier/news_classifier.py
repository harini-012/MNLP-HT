
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

MODEL_NAME = "distilbert-base-uncased"
MODEL_PATH = "./saved_model"
OUTPUT_PATH = "./training_results"

CATEGORIES = ["World", "Sports", "Business", "Sci/Tech"]

TRAIN_LIMIT = 2000
TEST_LIMIT = None
SEED = 42

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)
print("Loading AG News dataset...")

# ==========================================
# 2. LOAD DATASET
# ==========================================
dataset = load_dataset("fancyzhx/ag_news")

print("\nDataset information:")
print(dataset)

print("\nTraining samples:", len(dataset["train"]))
print("Testing samples:", len(dataset["test"]))

# Explore the four categories
train_df = pd.DataFrame(dataset["train"][:5000])
train_df["category"] = train_df["label"].map(
    dict(enumerate(CATEGORIES))
)

print("\nCategory distribution:")
print(train_df["category"].value_counts())

train_df["category"].value_counts().reindex(
    CATEGORIES
).plot(
    kind="bar",
    figsize=(8, 5),
    title="AG News Category Distribution"
)

plt.xlabel("Category")
plt.ylabel("Number of Articles")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# ==========================================
# 3. PREPARE TRAINING, VALIDATION AND TEST DATA
# ==========================================

train_data = dataset["train"].shuffle(seed=SEED)

if TRAIN_LIMIT is not None:
    train_data = train_data.select(
        range(min(TRAIN_LIMIT, len(train_data)))
    )

# Keep validation data separate from the official test data
splits = train_data.train_test_split(
    test_size=0.1,
    seed=SEED
)

train_data = splits["train"]
validation_data = splits["test"]

test_data = dataset["test"]

if TEST_LIMIT is not None:
    test_data = test_data.select(
        range(min(TEST_LIMIT, len(test_data)))
    )

print("\nTraining examples:", len(train_data))
print("Validation examples:", len(validation_data))
print("Test examples:", len(test_data))

# ==========================================
# 4. TOKENIZE NEWS ARTICLES
# ==========================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_data(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=192
    )

tokenized_train = train_data.map(
    tokenize_data,
    batched=True
)

tokenized_validation = validation_data.map(
    tokenize_data,
    batched=True
)

tokenized_test = test_data.map(
    tokenize_data,
    batched=True
)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

print("\nTokenization completed.")

print("\nExample input IDs:")
print(tokenized_train[0]["input_ids"][:20])

print("\nExample attention mask:")
print(tokenized_train[0]["attention_mask"][:20])

# ==========================================
# 5. LOAD PRETRAINED TRANSFORMER MODEL
# ==========================================

id2label = {
    i: label for i, label in enumerate(CATEGORIES)
}

label2id = {
    label: i for i, label in id2label.items()
}

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=4,
    id2label=id2label,
    label2id=label2id
)

print("\nTransformer model loaded.")

# ==========================================
# 6. DEFINE EVALUATION METRICS
# ==========================================

def compute_metrics(eval_pred):
    logits, true_labels = eval_pred

    predicted_labels = np.argmax(
        logits,
        axis=-1
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            average="weighted",
            zero_division=0
        )
    )

    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# ==========================================
# 7. CONFIGURE TRAINING
# ==========================================

training_args = TrainingArguments(
    output_dir=OUTPUT_PATH,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=100,
    report_to="none",
    seed=SEED,
    fp16=torch.cuda.is_available()
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_validation,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# ==========================================
# 8. FINE-TUNE THE MODEL
# ==========================================

print("\nStarting Transformer training...")
print("Training may take some time.")

trainer.train()

print("\nTraining completed.")

# ==========================================
# 9. EVALUATE ON OFFICIAL TEST DATA
# ==========================================

print("\nEvaluating on test dataset...")

evaluation_results = trainer.evaluate(
    eval_dataset=tokenized_test
)

print("\nEvaluation Metrics:")

for metric, value in evaluation_results.items():
    if isinstance(value, (int, float)):
        print(f"{metric}: {value:.4f}")

# ==========================================
# 10. CLASSIFICATION REPORT
# ==========================================

prediction_output = trainer.predict(
    tokenized_test
)

y_true = prediction_output.label_ids

y_pred = np.argmax(
    prediction_output.predictions,
    axis=1
)

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3],
        target_names=CATEGORIES,
        digits=4,
        zero_division=0
    )
)

# ==========================================
# 11. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1, 2, 3]
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CATEGORIES,
    yticklabels=CATEGORIES
)

plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")
plt.title("Transformer News Classifier Confusion Matrix")
plt.tight_layout()
plt.show()

# ==========================================
# 12. PREDICT A NEW NEWS ARTICLE
# ==========================================

def predict_news(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=192
    )

    inputs = {
        key: value.to(trainer.model.device)
        for key, value in inputs.items()
    }

    trainer.model.eval()

    with torch.no_grad():
        outputs = trainer.model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

    predicted_id = int(
        torch.argmax(probabilities).item()
    )

    return {
        "category": CATEGORIES[predicted_id],
        "confidence": float(
            probabilities[predicted_id].item()
        )
    }

# Test with the example from the assignment
example = (
    "Apple announces a new smartphone with "
    "advanced AI features."
)

result = predict_news(example)

print("\nExample Prediction:")
print("Article:", example)
print("Predicted Category:", result["category"])
print(f"Confidence: {result['confidence']:.2%}")

# ==========================================
# 13. ACCEPT USER INPUT
# ==========================================

while True:
    print("\nEnter a news article to classify.")
    print("Type 'exit' to stop.")

    article = input("News article: ").strip()

    if article.lower() == "exit":
        print("Program terminated.")
        break

    if not article:
        print("Please enter a non-empty article.")
        continue

    result = predict_news(article)

    print("\nPredicted Category:", result["category"])
    print(f"Confidence: {result['confidence']:.2%}")

# ==========================================
# 14. SAVE TRAINED MODEL AND TOKENIZER
# ==========================================

Path(MODEL_PATH).mkdir(
    parents=True,
    exist_ok=True
)

trainer.save_model(MODEL_PATH)
tokenizer.save_pretrained(MODEL_PATH)

print("\nModel saved successfully.")
print("Saved location:", Path(MODEL_PATH).resolve())
