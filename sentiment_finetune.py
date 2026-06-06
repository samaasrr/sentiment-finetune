!pip install transformers datasets peft torch scikit-learn -q

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
