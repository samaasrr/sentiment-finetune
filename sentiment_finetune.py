

import torch
print(torch.cuda.is_available())

from datasets import load_dataset
ds = load_dataset("stanfordnlp/sst2")
print(ds)

print(ds["train"][:5])

ds["train"] = ds["train"].select(range(5000))

from collections import Counter
print(Counter(ds["train"]["label"]))

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
print("Tokenizer loaded!")

def tokenize(batch):
    return tokenizer(batch["sentence"], truncation=True, padding="max_length", max_length=128)

ds = ds.map(tokenize, batched=True)
print("Done! Here's what the dataset looks like now:")
print(ds)

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
print(model)

from peft import get_peft_model, LoraConfig, TaskType

config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"]
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    eval_strategy="epoch",
    logging_steps=50,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"]
)

trainer.train()

from sklearn.metrics import f1_score, classification_report
import numpy as np

preds = trainer.predict(ds["validation"])
y_pred = np.argmax(preds.predictions, axis=-1)
y_true = ds["validation"]["label"]

print(classification_report(y_true, y_pred, target_names=["negative", "positive"]))

import matplotlib.pyplot as plt

log = trainer.state.log_history
train_loss = [x["loss"] for x in log if "loss" in x]

plt.figure(figsize=(8,4))
plt.plot(train_loss, color="#1B2A4A")
plt.title("Training Loss over Steps")
plt.xlabel("Steps (every 50)")
plt.ylabel("Loss")
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()
print("Saved!")

from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

fig, ax = plt.subplots(figsize=(6,5))
ConfusionMatrixDisplay.from_predictions(
    y_true,
    y_pred,
    display_labels=["Negative", "Positive"],
    colorbar=False,
    ax=ax
)
plt.title("Confusion Matrix — Validation Set")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Saved!")
