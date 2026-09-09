import torch
import pandas as pd
import numpy as np
from functions.evaluation import evaluate_model,evaluate_model_per_aspect
from functions.data_cleaning import load_data, load_data_single_label, LABEL_COL_NAME
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

NUM_LABELS = 3  # For multi-class classification (0: non-existent, 1: negative, 2: positive)
LABEL_COL_NAME = LABEL_COL_NAME

# Spliting the problem from a multi-label multi-output problem to a single-label multi-output problem for the evaluation of the model on each aspect separately

for aspect in LABEL_COL_NAME[1:]:
    print(f"Training model for the aspect:{aspect}")

    train_ds,test_ds, label_names = load_data_single_label(file_name=f"model_two_{aspect}_dataset.csv", seed=5, test_size=0.2,aspect=aspect)

    model_name = "readerbench/RoBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def tokenize_function(batch):
        return tokenizer(batch["content"], truncation=True)
    
    training_encoding = train_ds.map(tokenize_function, batched=True)
    testing_encoding = test_ds.map(tokenize_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=NUM_LABELS, problem_type="single_label_classification")
    # Evaluation metrics
    accuracy_metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        reviews, labels = eval_pred
        preds = np.argmax(reviews, axis=-1)  # Get the predicted class (0, 1, 2)
        labels = labels.astype(int)
        f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
        accuracy = (preds == labels).mean()
        return {"accuracy": accuracy, "f1": f1_macro}
    
    training_args = TrainingArguments(
        output_dir=f"./results/readerbench/RoBERT-base_{aspect}",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=15,
        weight_decay=0.1,
        report_to="none",
        load_best_model_at_end=True,
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
    with open(os.path.join(training_args.output_dir, "trainer_metrics_polarity.json"), "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"Training metrics saved to {training_args.output_dir}/trainer_metrics_polarity.json")

    trainer.evaluate(testing_encoding)

    predictions, labels, _ = trainer.predict(testing_encoding)
    preds = np.argmax(predictions, axis=-1)

    evaluate_model_per_aspect(y_true=labels, y_pred=preds, model_name=f"RoBERT-base_fine_tuning", aspect=aspect)