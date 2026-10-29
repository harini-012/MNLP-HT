
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
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ==================================================
# 1. CONFIGURATION
# ==================================================

MODEL_NAME = "bert-base-uncased"
MODEL_PATH = "saved_model"
OUTPUT_DIR = "training_results"
MAX_LENGTH = 128

# Smaller samples make training easier on a normal computer.
TRAIN_SAMPLES = 1200
VALIDATION_SAMPLES = 500
EPOCHS = 1
BATCH_SIZE = 8

print("=" * 60)
print("BERT CUSTOMER FEEDBACK SENTIMENT CLASSIFIER")
print("=" * 60)

print("\nLoading SST-2 sentiment dataset...")

# SST-2 contains sentences labeled as negative (0) or positive (1).
dataset = load_dataset("glue", "sst2")

train_data = (
    dataset["train"]
    .shuffle(seed=42)
    .select(range(min(TRAIN_SAMPLES, len(dataset["train"]))))
)

validation_data = (
    dataset["validation"]
    .select(range(min(VALIDATION_SAMPLES, len(dataset["validation"]))))
)

print("Training samples:", len(train_data))
print("Validation samples:", len(validation_data))

# ==================================================
# 2. LOAD BERT TOKENIZER
# ==================================================

print("\nLoading BERT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_data(examples):
    return tokenizer(
        examples["sentence"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


print("Tokenizing training data...")
train_data = train_data.map(tokenize_data, batched=True)

print("Tokenizing validation data...")
validation_data = validation_data.map(tokenize_data, batched=True)

# Rename the dataset label column for Hugging Face Trainer.
train_data = train_data.rename_column("label", "labels")
validation_data = validation_data.rename_column("label", "labels")

# BERT requires input_ids and attention_mask.
columns = ["input_ids", "attention_mask", "labels"]

train_data.set_format(type="torch", columns=columns)
validation_data.set_format(type="torch", columns=columns)

# ==================================================
# 3. LOAD BERT CLASSIFICATION MODEL
# ==================================================

print("\nLoading BERT sequence classification model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)

# ==================================================
# 4. EVALUATION METRICS
# ==================================================

def compute_metrics(eval_prediction):
    logits = eval_prediction.predictions
    labels = eval_prediction.label_ids

    # Some models may return predictions as a tuple.
    if isinstance(logits, tuple):
        logits = logits[0]

    predicted_labels = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predicted_labels),
        "precision": precision_score(
            labels, predicted_labels, zero_division=0
        ),
        "recall": recall_score(
            labels, predicted_labels, zero_division=0
        ),
        "f1": f1_score(
            labels, predicted_labels, zero_division=0
        ),
    }


# ==================================================
# 5. TRAINING CONFIGURATION
# ==================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=25,
    report_to="none",
    use_cpu=not torch.cuda.is_available(),
    seed=42,
)

# ==================================================
# 6. TRAIN THE MODEL
# ==================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=validation_data,
    compute_metrics=compute_metrics,
)

print("\nStarting BERT fine-tuning...")
print("Training may take some time, especially on a CPU.")

trainer.train()

# ==================================================
# 7. EVALUATE THE MODEL
# ==================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

evaluation_results = trainer.evaluate()

for metric, value in evaluation_results.items():
    if isinstance(value, (int, float)):
        print(f"{metric}: {value:.4f}")
    else:
        print(f"{metric}: {value}")

# ==================================================
# 8. CLASSIFICATION REPORT AND CONFUSION MATRIX
# ==================================================

print("\nGenerating detailed classification report...")

prediction_output = trainer.predict(validation_data)

logits = prediction_output.predictions
if isinstance(logits, tuple):
    logits = logits[0]

predicted_labels = np.argmax(logits, axis=-1)
true_labels = prediction_output.label_ids

print(
    classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=["Negative", "Positive"],
        zero_division=0,
    )
)

print("Confusion matrix:")
print(confusion_matrix(true_labels, predicted_labels, labels=[0, 1]))

# ==================================================
# 9. SAVE MODEL AND TOKENIZER
# ==================================================

print("\nSaving the trained model...")

os.makedirs(MODEL_PATH, exist_ok=True)

trainer.save_model(MODEL_PATH)
tokenizer.save_pretrained(MODEL_PATH)

print(f"Model and tokenizer saved to: {os.path.abspath(MODEL_PATH)}")

# ==================================================
# 10. PREDICT CUSTOMER FEEDBACK
# ==================================================

def predict_sentiment(text):
    """Return the predicted sentiment and its confidence percentage."""

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )

    # Move inputs to the same device as the model.
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)

    predicted_class = torch.argmax(probabilities, dim=-1).item()

    sentiment = "Positive" if predicted_class == 1 else "Negative"
    confidence = probabilities[0, predicted_class].item() * 100

    return sentiment, confidence


# ==================================================
# 11. TEST EXAMPLE FEEDBACK
# ==================================================

print("\n" + "=" * 60)
print("EXAMPLE PREDICTIONS")
print("=" * 60)

examples = [
    "The product is excellent and works perfectly.",
    "I am very disappointed with this purchase.",
    "The customer service was helpful and friendly.",
    "The quality is terrible and the product broke.",
    "The experience was okay, nothing special.",
]

for feedback in examples:
    sentiment, confidence = predict_sentiment(feedback)

    print(f"\nFeedback: {feedback}")
    print(f"Predicted sentiment: {sentiment}")
    print(f"Confidence: {confidence:.2f}%")

# ==================================================
# 12. INTERACTIVE PREDICTION
# ==================================================

print("\n" + "=" * 60)
print("CUSTOMER FEEDBACK SENTIMENT CLASSIFIER")
print("Enter feedback to analyze it.")
print("Type 'exit' to stop.")
print("=" * 60)

while True:
    try:
        feedback = input("\nEnter customer feedback: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nProgram closed.")
        break

    if feedback.lower() == "exit":
        print("Program closed.")
        break

    if not feedback:
        print("Please enter some feedback.")
        continue

    sentiment, confidence = predict_sentiment(feedback)

    print(f"Predicted sentiment: {sentiment}")
    print(f"Confidence: {confidence:.2f}%")
