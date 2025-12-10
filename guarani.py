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
import subprocess
import unicodedata

# Preprocess text with improved functionality
def preprocess_text(text, lowercase=True, remove_special_chars=True, normalize_unicode=True):
    if normalize_unicode:
        text = unicodedata.normalize("NFKC", text)
    if lowercase:
        text = text.lower()
    if remove_special_chars:
        # Retain essential punctuation but clean unnecessary symbols
        text = re.sub(r"[^\w\s.,'\"áéíóúñãẽĩõũ]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()  # Normalize spaces
    return text

# Merge Guarani digraphs and nasal vowels
def merge_digraphs(text):
    digraphs = {"c h": "ch", "m b": "mb", "n g": "ng"}  # Add spaces for tokenized digraphs
    for key, value in digraphs.items():
        text = text.replace(key, value)
    return text

# Analyze and visualize text lengths and ratios
def analyze_lengths(source_lengths, target_lengths, length_ratios, threshold=2.5):
    plt.figure(figsize=(12, 6))
    plt.hist(source_lengths, bins=30, alpha=0.5, label="Source Lengths (Spanish)", color="blue")
    plt.hist(target_lengths, bins=30, alpha=0.5, label="Target Lengths (Guarani)", color="green")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.title("Distribution of Text Lengths in Source and Target")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.hist(length_ratios, bins=30, color="purple", alpha=0.7, label="Length Ratios (Target/Source)")
    plt.axvline(x=1 / threshold, color="red", linestyle="--", label="Lower Threshold")
    plt.axvline(x=threshold, color="red", linestyle="--", label="Upper Threshold")
    plt.xlabel("Length Ratio")
    plt.ylabel("Frequency")
    plt.title("Distribution of Length Ratios")
    plt.legend()
    plt.grid(True)
    plt.show()

# Load paired dataset with improved preprocessing
def load_and_preprocess_paired_data(source_file, target_file, length_ratio_threshold=2.5):
    with open(source_file, "r", encoding="utf-8") as src, open(target_file, "r", encoding="utf-8") as tgt:
        source_lines = src.readlines()
        target_lines = tgt.readlines()

    source_lengths = []
    target_lengths = []
    length_ratios = []
    preprocessed_data = []

    for src, tgt in zip(source_lines, target_lines):
        src_clean = preprocess_text(src)
        tgt_clean = preprocess_text(tgt)
        tgt_clean = merge_digraphs(tgt_clean)
        if not src_clean or not tgt_clean:
            continue
        source_len = len(src_clean.split())
        target_len = len(tgt_clean.split())
        if source_len == 0:
            continue

        length_ratio = target_len / source_len
        source_lengths.append(source_len)
        target_lengths.append(target_len)
        length_ratios.append(length_ratio)

        if 1 / length_ratio_threshold <= length_ratio <= length_ratio_threshold:
            preprocessed_data.append({"source_text": src_clean, "target_text": tgt_clean})

    analyze_lengths(source_lengths, target_lengths, length_ratios, length_ratio_threshold)
    print(f"Total Pairs: {len(source_lines)}")
    print(f"Valid Pairs After Filtering: {len(preprocessed_data)}")
    print(f"Average Source Length: {sum(source_lengths) / len(source_lengths):.2f}")
    print(f"Average Target Length: {sum(target_lengths) / len(target_lengths):.2f}")
    return preprocessed_data

# Load the paired training and validation datasets
train_data = load_and_preprocess_paired_data("data/combined_train.es", "data/combined_train.gn")
val_data = load_and_preprocess_paired_data("data/combined_val.es", "data/combined_val.gn")

# Convert to Hugging Face Datasets
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# Tokenizer setup
tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50")
tokenizer.src_lang = "es_XX"
tokenizer.tgt_lang = "gn_XX"

# Add Guarani-specific tokens if necessary
guarani_special_tokens = ["\u0303", "ã", "ẽ", "ĩ", "õ", "ũ", "ch", "mb", "ng"]
tokenizer.add_tokens(guarani_special_tokens)
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
model.resize_token_embeddings(len(tokenizer))

# Data preprocessing for Hugging Face
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

# Define compute_metrics function
def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute BLEU score
    bleu = BLEU()
    bleu_score = bleu.corpus_score(decoded_preds, [decoded_labels])
    return {"bleu": bleu_score.score}

# Define the directory to save results
results_dir = "./results123"
os.makedirs(results_dir, exist_ok=True)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=results_dir,
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=500,
    save_steps=500,
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=15,
    predict_with_generate=True,
    fp16=True,
    save_total_limit=3,
    logging_dir="./logs",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    lr_scheduler_type="cosine",
    warmup_steps=800,
    weight_decay=0.01,
    label_smoothing_factor=0.2,
)

# Trainer setup
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained(results_dir)
tokenizer.save_pretrained(results_dir)

# Generate predictions
predictions, labels, _ = trainer.predict(tokenized_val)
decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

# Save predictions and references for evaluation
pred_file = os.path.join(results_dir, "predictions.txt")
ref_file = os.path.join(results_dir, "references.txt")

with open(pred_file, "w") as f:
    f.write("\n".join(decoded_preds))

with open(ref_file, "w") as f:
    f.write("\n".join(decoded_labels))

# Call external evaluation script
evaluation_script = "./evaluate.py"  # Path to your evaluation script
subprocess.run(
    ["python", evaluation_script, "--system_output", pred_file, "--gold_reference", ref_file, "--detailed_output"]
)

# Plot learning curve
training_metrics = [log["loss"] for log in trainer.state.log_history if "loss" in log]
eval_metrics = [log["eval_loss"] for log in trainer.state.log_history if "eval_loss" in log]

plt.figure(figsize=(10, 6))
plt.plot(training_metrics, label="Training Loss", marker="o")
plt.plot(eval_metrics, label="Validation Loss", marker="o")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Learning Curve")
plt.legend()
plt.grid(True)

# Save learning curve
learning_curve_file = os.path.join(results_dir, "learning_curve.png")
plt.savefig(learning_curve_file)
print(f"Learning curve plot saved to {learning_curve_file}")
