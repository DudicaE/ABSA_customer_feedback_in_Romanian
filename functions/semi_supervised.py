# File which will contain the semi-supervised learning function 

# Using the RoBert model for semi-supervised learning on the 4000 reviews dataset. The model will be trained on 4000 reviews out of the 13000 remaining reviews due to technical limitations of memory and processing power. The model accuracy will use

#Import the 4000 reviews from the file 
from functions.prediction_class import predict_reviews
import torch
import pandas as pd
import numpy as np
from functions.evaluation import evaluate_model,evaluate_model_per_aspect
from functions.data_cleaning import load_data, load_data_single_label,LABEL_COL_NAME
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

def save_4000_reviews(input_file:str, output_file:str):

    df = pd.read_csv(input_file, sep=",")
    # Selection of the 4000 reviews to be used for training the model. The selection will be based on the star rating of the reviews based on the distribution of the star ratings in the dataset. 
    positive_reviews_4_stars = df[df['starRating'] == 4].sample(n=500, random_state=1)
    positive_reviews_5_stars = df[df['starRating'] == 5].sample(n=1500, random_state=1)
    negative_reviews_1_star = df[df['starRating'] == 1].sample(n=1500, random_state=1)
    negative_reviews_2_stars = df[df['starRating'] == 2].sample(n=500, random_state=1)
    selected_reviews = pd.concat([positive_reviews_4_stars, positive_reviews_5_stars, negative_reviews_1_star, negative_reviews_2_stars], ignore_index=True)
    selected_reviews.to_csv(output_file, index=False,sep=";")




# Using RoBert trained on the golden standard dataset to predict the aspect and polarity of the 4000 reviews. The model will be trained on 4000 reviews out of the 13000 remaining reviews due to technical limitations of memory and processing power. The model accuracy will use the golden standard dataset as a reference.
def predict_4000_reviews(input_file:str, output_file:str):
    predict_reviews(
        file_name=input_file,
        output_file=output_file,
        aspect_model_path="./results/readerbench/robert-base_fine_tuning/checkpoint-500", 
        polarity_model_path="./results/readerbench/robert-base_", 
        label_names=["content", "design", "functionalitate", "livrare_ambalaj", "calitate_pret", "satisfactie_generala"]
        )


# Fine-tuning the RoBert model on the 4000 reviews dataset. The model will be trained on 4000 reviews out of the 13000 remaining reviews due to technical limitations of memory and processing power. The model accuracy will use the golden standard dataset as a reference.
def fine_tune_robert_on_4000_reviews_model_one(input_file:str):

    train_ds,test_ds, label_names = load_data(file_name=input_file, seed=5, test_size=0.2)

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
        output_dir="./results/semi_supervised/robert-base_fine_tuning",
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
    thresholds = [0.5, 0.5, 0.3, 0.6, 0.5]  # Ajust thresholds for each label based on validation performance  since label 2 is more imbalanced, we set a lower threshold for it to improve recall.

    pred_labels = (probs > thresholds).astype(int)

    true_labels = predictions.label_ids.astype(int)

    # Create directory for saving results if it doesn't exist
    os.makedirs("./results/semi_supervised/robert_base_fine_tuning", exist_ok=True),

    evaluate_model(
        y_true=true_labels,
        y_pred=pred_labels,
        model_name="semi_supervised_robert_base_fine_tuning"
    )



def fine_tune_robert_on_4000_reviews_model_two():
    
    # Spliting the problem from a multi-label multi-output problem to a single-label multi-output problem for the evaluation of the model on each aspect separately

    for aspect in LABEL_COL_NAME[1:]:
        print(f"Training model for the aspect:{aspect}")

        train_ds,test_ds, label_names = load_data_single_label(file_name=f"semi_supervised/model_two_{aspect}_dataset.csv", seed=5, test_size=0.2,aspect=aspect)

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
            output_dir=f"./results/semi_supervised/RoBERT-base_{aspect}",
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

        evaluate_model_per_aspect(y_true=labels, y_pred=preds, model_name=f"semi_supervised_RoBERT_fine_tuning", aspect=aspect)
        