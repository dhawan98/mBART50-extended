import os
import re
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (
    MBart50Tokenizer,
    MBartForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from sacrebleu.metrics import BLEU
import unicodedata
from collections import Counter

# Debug logging function
def debug_log(message):
    print(f"[DEBUG]: {message}")

# Enhanced text preprocessing
def preprocess_text(text, lowercase=True, remove_special_chars=True, normalize_unicode=True):
    if normalize_unicode:
        text = unicodedata.normalize("NFKC", text)
    if lowercase:
        text = text.lower()
    if remove_special_chars:
        text = re.sub(r"[^\w\s.,'\"áéíóúñäëïöü]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Aymara-specific linguistic optimizations
def optimize_aymara_text(text):
    replacements = {
        "c h": "ch",
        "m b": "mb",
        "n g": "ng",
        "j a t x": "jatx",
        "u k a t x": "ukatx",
        "ñ": "ñ",
    }
    suffixes = ["-x", "-niwa", "-ta", "-kama", "-wa"]
    for key, value in replacements.items():
        text = text.replace(key, value)
    for suffix in suffixes:
        text = re.sub(rf"({suffix})", r"\1", text)
    return text

# Rare token replacement
def replace_rare_tokens(text, rare_tokens):
    tokens = text.split()
    replaced_tokens = [token if token not in rare_tokens else "<rare>" for token in tokens]
    return " ".join(replaced_tokens)

# Clean and filter sentence pairs
def clean_and_filter_data(source_lines, target_lines, length_ratio_min=0.7, length_ratio_max=1.3, rare_threshold=5):
    valid_pairs = []
    source_lengths, target_lengths = [], []
    source_token_counts = Counter()
    target_token_counts = Counter()

    for src, tgt in zip(source_lines, target_lines):
        src_clean = preprocess_text(src)
        tgt_clean = preprocess_text(tgt)
        tgt_clean = optimize_aymara_text(tgt_clean)

        if not src_clean or not tgt_clean:
            continue

        src_tokens = src_clean.split()
        tgt_tokens = tgt_clean.split()
        source_token_counts.update(src_tokens)
        target_token_counts.update(tgt_tokens)

        source_len = len(src_tokens)
        target_len = len(tgt_tokens)
        if source_len == 0 or target_len == 0:
            continue
        length_ratio = target_len / source_len
        if length_ratio_min <= length_ratio <= length_ratio_max:
            valid_pairs.append({"source_text": src_clean, "target_text": tgt_clean})
            source_lengths.append(source_len)
            target_lengths.append(target_len)

    debug_log(f"Valid pairs after filtering: {len(valid_pairs)}")

    # Identify rare tokens
    rare_source_tokens = {token for token, count in source_token_counts.items() if count < rare_threshold}
    rare_target_tokens = {token for token, count in target_token_counts.items() if count < rare_threshold}
    debug_log(f"Rare source tokens: {len(rare_source_tokens)}")
    debug_log(f"Rare target tokens: {len(rare_target_tokens)}")

    # Replace rare tokens
    for pair in valid_pairs:
        pair["source_text"] = replace_rare_tokens(pair["source_text"], rare_source_tokens)
        pair["target_text"] = replace_rare_tokens(pair["target_text"], rare_target_tokens)

    return valid_pairs

# Load and preprocess data
def load_data(source_file, target_file, length_ratio_min=0.7, length_ratio_max=1.3):
    with open(source_file, "r", encoding="utf-8") as src, open(target_file, "r", encoding="utf-8") as tgt:
        source_lines = src.readlines()
        target_lines = tgt.readlines()
    return clean_and_filter_data(source_lines, target_lines, length_ratio_min, length_ratio_max)

# Load training and validation datasets
train_data = load_data("data/aymara-spanish/train.es", "data/aymara-spanish/train.aym")
val_data = load_data("data/aymara-spanish/dev.es", "data/aymara-spanish/dev.aym")

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# Tokenizer setup
tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50")
tokenizer.src_lang = "es_XX"
tokenizer.tgt_lang = "aym_XX"

# Add Aymara-specific tokens
aymara_special_tokens = ["\u0303", "ä", "ë", "ï", "ö", "ü", "ch", "mb", "ng", "jatx", "ukatx", "<rare>"]
tokenizer.add_tokens(aymara_special_tokens)
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50", dropout=0.3, attention_dropout=0.3)
model.resize_token_embeddings(len(tokenizer))

# Preprocessing function for Hugging Face datasets
def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["source_text"], max_length=128, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        examples["target_text"], max_length=128, truncation=True, padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=["source_text", "target_text"])
tokenized_val = val_dataset.map(preprocess_function, batched=True, remove_columns=["source_text", "target_text"])

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results_aym3",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=500,
    save_steps=500,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,
    num_train_epochs=20,
    predict_with_generate=True,
    fp16=True,
    save_total_limit=3,
    logging_dir="./logs_aym",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    lr_scheduler_type="cosine",
    warmup_steps=800,
    weight_decay=0.01,
    label_smoothing_factor=0.2,
    report_to="none",
)

# Trainer setup
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=lambda p: {"bleu": BLEU().corpus_score(
        tokenizer.batch_decode(p.predictions, skip_special_tokens=True),
        [tokenizer.batch_decode(p.label_ids, skip_special_tokens=True)]
    ).score},
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Train the model
trainer.train()

# Save predictions and references
predictions, labels, _ = trainer.predict(tokenized_val)
decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

results_dir = "./results_aym3"
os.makedirs(results_dir, exist_ok=True)
with open(os.path.join(results_dir, "predictions.txt"), "w") as f:
    f.write("\n".join(decoded_preds))
with open(os.path.join(results_dir, "references.txt"), "w") as f:
    f.write("\n".join(decoded_labels))

# Learning curve
training_logs = trainer.state.log_history
loss_values = [log["loss"] for log in training_logs if "loss" in log]
bleu_values = [log["eval_bleu"] for log in training_logs if "eval_bleu" in log]
epochs = list(range(1, len(loss_values) + 1))

plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_values, label="Training Loss", marker="o")
plt.plot(range(1, len(bleu_values) + 1), bleu_values, label="Validation BLEU", marker="o")
plt.xlabel("Epochs")
plt.ylabel("Metric")
plt.title("Learning Curve: Loss and BLEU")
plt.legend()
plt.grid(True)

# Save learning curve
learning_curve_file = os.path.join(results_dir, "learning_curve.png")
plt.savefig(learning_curve_file)
print(f"Learning curve saved to {learning_curve_file}")
