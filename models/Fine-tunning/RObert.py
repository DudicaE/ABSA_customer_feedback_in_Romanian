import torch
import pandas as pd
import numpy as np
from functions.evaluation import evaluate_model,evaluate_model_per_aspect
from functions.data_cleaning import load_data, load_data_single_label
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)
import evaluate
from scipy.special import expit  # sigmoid
from sklearn.metrics import f1_score, precision_score, recall_score
import os
import json

train_ds,test_ds, label_names = load_data(file_name="model_one_dataset.csv", seed=5, test_size=0.2)

model_name = "readerbench/RoBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(batch):
    return tokenizer(batch["content"], truncation=True)

training_encoding = train_ds.map(tokenize_function, batched=True)
testing_encoding = test_ds.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

num_labels = len(label_names)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, problem_type="multi_label_classification")

# Evaluation metrics
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    text,labels = eval_pred
    probs = expit(text)                  
    preds = (probs > 0.5).astype(int)     
    labels = labels.astype(int)

    f1_micro = f1_score(labels, preds, average="micro", zero_division=0)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)
    return {"f1_micro": f1_micro, "f1_macro": f1_macro, "precision": precision, "recall": recall}


training_args = TrainingArguments(  
    output_dir="./results/readerbench/robert-base_fine_tuning",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=15,
    weight_decay=0.1,
    report_to="none",
    fp16=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_encoding,
    eval_dataset=testing_encoding,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
    )

trainer.train()
print("Best checkpoint:", trainer.state.best_model_checkpoint)
print("Best metric:", trainer.state.best_metric)
trainer.save_model(training_args.output_dir)
trainer.save_state()
with open(os.path.join(training_args.output_dir, "trainer_metrics3.json"), "w", encoding="utf-8") as f:
    json.dump(trainer.state.log_history, f, indent=2)
print(f"Training metrics saved to {training_args.output_dir}/trainer_metrics3.json")

trainer.evaluate(testing_encoding)

predictions = trainer.predict(testing_encoding)

probs = expit(predictions.predictions)
thresholds = [0.5, 0.5, 0.15, 0.6, 0.5]  # Ajust thresholds for each label based on validation performance  since label 2 is more imbalanced, we set a lower threshold for it to improve recall.

pred_labels = (probs > thresholds).astype(int)

true_labels = predictions.label_ids.astype(int)

# Create directory for saving results if it doesn't exist
os.makedirs("./results/readerbench/robert-base_fine_tuning", exist_ok=True),

evaluate_model(
    y_true=true_labels,
    y_pred=pred_labels,
    model_name="robert-base_fine_tuning_3"
)