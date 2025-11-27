# app/model/train_finetune_embeddings.py

import pandas as pd
import random
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("unified_courses_v1.csv")

def safe(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

# Extract available fields
titles = df["course_title"].apply(safe).tolist()
descriptions = df["description"].apply(safe).tolist() if "description" in df.columns else ["" for _ in range(len(df))]
categories = df["category"].apply(safe).tolist()
subcats = df["subcategory"].apply(safe).tolist()

# what_you_will_learn (if exists)
wyl_items = []
if "what_you_will_learn" in df.columns:
    for x in df["what_you_will_learn"]:
        if isinstance(x, list):
            wyl_items.extend([str(i) for i in x])
        else:
            wyl_items.append(safe(x))

# -----------------------------
# 2. Build training samples
# -----------------------------
train_samples = []

for idx, row in df.iterrows():
    title = safe(row["course_title"])
    desc = safe(row["description"]) if "description" in df.columns else ""
    cat = safe(row["category"])
    sub = safe(row["subcategory"])

    # positive pairs
    train_samples.append(InputExample(texts=[title, desc], label=1.0))
    train_samples.append(InputExample(texts=[title, cat], label=1.0))
    train_samples.append(InputExample(texts=[title, sub], label=1.0))

    if "what_you_will_learn" in df.columns and isinstance(row["what_you_will_learn"], list):
        for w in row["what_you_will_learn"]:
            train_samples.append(InputExample(texts=[title, str(w)], label=1.0))

# negative pairs
for _ in range(50000):
    i, j = random.sample(range(len(df)), 2)
    title = titles[i]
    random_text = random.choice([descriptions[j], categories[j], subcats[j]])
    train_samples.append(InputExample(texts=[title, random_text], label=0.0))

# shuffle
random.shuffle(train_samples)

# -----------------------------
# 3. Load base model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# 4. Train setup
# -----------------------------
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=32)
train_loss = losses.CosineSimilarityLoss(model)

num_epochs = 3
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

# -----------------------------
# 5. Train
# -----------------------------
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    show_progress_bar=True,
)

# -----------------------------
# 6. Save model
# -----------------------------
save_path = "model/miniLM_finetuned_udemy"
model.save(save_path)
print("Fine-tuned model saved at:", save_path)
