
import os
import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report

# -------------------------------
# 1. Configuration
# -------------------------------

MODEL_NAME = "distilbert-base-uncased"
MODEL_PATH = "saved_model"

print("Loading dataset...")

# SST-2 sentiment dataset
dataset = load_dataset("glue", "sst2")

# Use smaller samples to make training easier on a normal computer
train_data = dataset["train"].shuffle(seed=42).select(range(1200))
validation_data = dataset["validation"].select(range(500))

# -------------------------------
# 2. Tokenization
# -------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_data(examples):
    return tokenizer(
        examples["sentence"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )


print("Tokenizing dataset...")

train_data = train_data.map(tokenize_data, batched=True)
validation_data = validation_data.map(tokenize_data, batched=True)

train_data = train_data.rename_column("label", "labels")
validation_data = validation_data.rename_column("label", "labels")

columns = ["input_ids", "attention_mask", "labels"]

train_data.set_format(type="torch", columns=columns)
validation_data.set_format(type="torch", columns=columns)

# -------------------------------
# 3. Load DistilBERT
# -------------------------------

print("Loading DistilBERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)

# -------------------------------
# 4. Evaluation metrics
# -------------------------------

def compute_metrics(prediction):
    logits, labels = prediction
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
    }


# -------------------------------
# 5. Training configuration
# -------------------------------

training_args = TrainingArguments(
    output_dir="training_results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=50,
    report_to="none",
    use_cpu=not torch.cuda.is_available(),
)

# -------------------------------
# 6. Train the model
# -------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=validation_data,
    compute_metrics=compute_metrics,
)

print("\nStarting model training...")
trainer.train()

# -------------------------------
# 7. Evaluate the model
# -------------------------------

print("\nEvaluating model...")

results = trainer.evaluate()
print("\nEvaluation results:")
for key, value in results.items():
    print(f"{key}: {value}")

# -------------------------------
# 8. Detailed classification report
# -------------------------------

print("\nGenerating classification report...")

predictions = trainer.predict(validation_data)
predicted_labels = np.argmax(predictions.predictions, axis=-1)
true_labels = predictions.label_ids

print(
    classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=["Negative", "Positive"],
        zero_division=0,
    )
)

# -------------------------------
# 9. Save trained model
# -------------------------------

print("\nSaving trained model...")

os.makedirs(MODEL_PATH, exist_ok=True)
trainer.save_model(MODEL_PATH)
tokenizer.save_pretrained(MODEL_PATH)

print(f"Model saved successfully in '{MODEL_PATH}'")

# -------------------------------
# 10. Predict custom feedback
# -------------------------------

def predict_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()

    sentiment = "Positive" if predicted_class == 1 else "Negative"
    confidence = probabilities[0][predicted_class].item() * 100

    return sentiment, confidence


# -------------------------------
# 11. Test example feedback
# -------------------------------

print("\nExample predictions:")

examples = [
    "The product is excellent and works perfectly.",
    "I am very disappointed with this purchase.",
    "The customer service was helpful and friendly.",
    "The quality is terrible and the product broke.",
]

for feedback in examples:
    sentiment, confidence = predict_sentiment(feedback)
    print(f"\nFeedback: {feedback}")
    print(f"Sentiment: {sentiment}")
    print(f"Confidence: {confidence:.2f}%")

# -------------------------------
# 12. Interactive prediction
# -------------------------------

print("\nCustomer Feedback Sentiment Classifier")
print("Type 'exit' to stop.")

while True:
    feedback = input("\nEnter customer feedback: ").strip()

    if feedback.lower() == "exit":
        print("Program closed.")
        break

    if not feedback:
        print("Please enter some feedback.")
        continue

    sentiment, confidence = predict_sentiment(feedback)

    print(f"Predicted sentiment: {sentiment}")
    print(f"Confidence: {confidence:.2f}%")
