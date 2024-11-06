from datasets import load_dataset
from transformers import (
    MBart50Tokenizer,
    MBartForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
import torch

# Check GPU availability

import torch
print("CUDA Available:", torch.cuda.is_available())

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load dataset files
data_files = {
    "train": ["guarani-spanish/train.es", "guarani-spanish/train.gn"],
    "validation": ["guarani-spanish/dev.es", "guarani-spanish/dev.gn"]
}
dataset = load_dataset("text", data_files=data_files)

# Initialize tokenizer and model, and move model to GPU if available
tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50")
tokenizer.src_lang = "es_XX"
tokenizer.tgt_lang = "gn_XX"

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50").to(device)

# Define preprocessing function
def preprocess_function(examples):
    model_inputs = tokenizer(
        examples['text'], max_length=64, truncation=True, padding="max_length"
    )
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

# Apply preprocessing and confirm columns
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["text"])
print("Column Names for Train Dataset:", tokenized_datasets["train"].column_names)
print("Column Names for Validation Dataset:", tokenized_datasets["validation"].column_names)

# Data collator with padding on max_length
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="max_length")

# Define training arguments, enable fp16 for mixed precision
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,  # Enable mixed precision for GPU
    remove_unused_columns=False
)

# Initialize Trainer with explicit device handling
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Start training
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print("Evaluation Results:", results)
