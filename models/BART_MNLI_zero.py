import pandas as pd 
import torch
from transformers import  pipeline
from functions.data_cleaning import load_data,load_data_single_label
import time
from functions.evaluation import evaluate_model,evaluate_model_per_aspect


def zero_shot_BART(file_name:str):
    """Function which creates the structure for BART zero-shot classification which is not supporting any prompt template. The function takes as input the review to be classified and evaluated the model response to obtain the predicted labels. The predicted labels are returned as a list of integers and placed in to the evaluation function."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    train_data, test_data, label_columns = load_data(file_name=file_name, seed=5, test_size=0.2)

    # Load the model and tokenizer
    model_name = "facebook/bart-large-mnli"

    classifier = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device,
        # hypothesis_template=f"Această recenzie menționează una dintre aceste aspecte {label_columns}."
    )
    print('Model and tokenizer loaded')

    thresholds = {
        "design": 0.5,
        "functionalitate": 0.5,
        "livrare_ambalaj": 0.5,
        "calitate_pret": 0.5,
        "satisfactie_generala": 0.5
    }

    all_predictions = []
    print("Starting zero-shot classification...")
    for i in range(len(test_data)):
        review_text = str(test_data["content"][i])

        result = classifier(
            review_text,
            candidate_labels=label_columns,
            multi_label=True
        )
        score_dict = {
            label: score
            for label, score in zip(result["labels"], result["scores"])
        }
        pred = [
            int(score_dict["design"] >= thresholds["design"]),
            int(score_dict["functionalitate"] >= thresholds["functionalitate"]),
            int(score_dict["livrare_ambalaj"] >= thresholds["livrare_ambalaj"]),
            int(score_dict["calitate_pret"] >= thresholds["calitate_pret"]),
            int(score_dict["satisfactie_generala"] >= thresholds["satisfactie_generala"]),
        ]
        all_predictions.append(pred)
        print(f"Review: {review_text}")
        print(f"Scores: {score_dict}")
        print(f"Predicted labels: {pred}")

    # Evaluate the model
    print('Evaluating the model...')
    y_true = test_data['label']
    print("Evaluation metrics for the model:")
    evaluate_model(y_true, all_predictions, model_name=model_name.split("/")[-1] + "_model_zero_shot")



def zero_shot_BART_two(file_name:str, aspect:str):
    """Function which performs zero-shot classification using the BART model. The function takes as input the review to be classified and evaluated the model response to obtain the predicted labels. The predicted labels are returned as a list of integers and placed in to the evaluation function. This is a second version of the function which uses a different approach for the second model dataset and avoid the multi-label multi output which cannot be evaluated using the same metrics as the first model dataset."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    train_data, test_data, label_columns = load_data_single_label(file_name=file_name,aspect=aspect, seed=5, test_size=0.2)

    # Load the model and tokenizer
    model_name = "facebook/bart-large-mnli"

    classifier = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device
    )
    print('Model and tokenizer loaded')

    # Polarity thresholds for each label
    thresholds = {
        "non-present": 0.5,
        "negative": 0.5,
        "positive": 0.5
    }

    all_predictions = []
    print("Starting zero-shot classification...")
    for i in range(len(test_data)):
        review_text = str(test_data["content"][i])

        result = classifier(
            review_text,
            candidate_labels=["non-present", "negative", "positive"]
        )
        score_dict = {
            label: score for label, score in zip(result["labels"], result["scores"])
        }
         # Get from the dictonary the highest scoring label that is above the threshold else non-present
        
        highest_label = max(score_dict, key=score_dict.get)
        if score_dict[highest_label] >= thresholds[highest_label]:
            pred = highest_label
        else:
            pred = "non-present"
        
        pred_mapping = {
            "non-present": 0,
            "negative": 1,
            "positive": 2
        }
        pred = pred_mapping[pred]
        all_predictions.append(pred)
        print(f"Review: {review_text}")
        print(f"Scores: {score_dict}")
        print(f"Predicted labels: {pred}")
       
    # Evaluate the model
    print('Evaluating the model...')
    y_true = test_data['label']
    print("Evaluation metrics for the model:")
    evaluate_model_per_aspect(y_true, all_predictions, model_name=model_name.split("/")[-1] + "_model_zero_shot",aspect=aspect)