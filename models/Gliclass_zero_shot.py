import pandas as pd 
import torch
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer, pipeline
from functions.data_cleaning import load_data,load_data_single_label
import time
from functions.evaluation import evaluate_model,evaluate_model_per_aspect

def gliclass_zero_shot_classification(file_name:str,model_name:str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load the model and tokenizer 
    train_data, test_data, label_columns = load_data(file_name=file_name, seed=5, test_size=0.2)
    model = GLiClassModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained("knowledgator/gliclass-multilang-mini")
    
    print('Model and tokenizer loaded')

    pipeline = ZeroShotClassificationPipeline(model, tokenizer, classification_type='multi-label', device=device)
     # Zero-shot classification
    print('Starting zero-shot classification...')

    print('Model and tokenizer loaded')
    all_predictions = []
    print("Starting zero-shot classification...")
    for i in range(len(test_data)):
        review_text = str(test_data["content"][i])

        result = pipeline(
            review_text,
            label_columns
        )[0]
        print(result)
        structure = [0,0,0,0,0] 
        for r in result:
            
            if r['label'] in label_columns:
                index = label_columns.index(r['label'])
                structure[index] = 1
        
        all_predictions.append(structure)
    # Evaluate the model
    print('Evaluating the model...')
    y_true = test_data['label']
    print("Evaluation metrics for the model:")
    evaluate_model(y_true, all_predictions, model_name=model_name.split("/")[-1] + "_model_zero_shot")


def gliclass_zero_shot_classification_polarity(file_name:str,model_name:str, aspect:str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load the model and tokenizer 
    train_data, test_data, label_columns = load_data_single_label(file_name=file_name, seed=5, test_size=0.2,aspect=aspect)
    model = GLiClassModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained("knowledgator/gliclass-multilang-mini")
    
    print('Model and tokenizer loaded')

    pipeline = ZeroShotClassificationPipeline(model, tokenizer, classification_type='single-label', device=device)
     # Zero-shot classification
    print('Starting zero-shot classification...')

    print('Model and tokenizer loaded')
    all_predictions = []
    print("Starting zero-shot classification...")
    for i in range(len(test_data)):
        review_text = str(test_data["content"][i])

        result = pipeline(
            review_text,
            ["non-present", "negative", "positive"]
        )[0]
        print(result)
        mapping = {
            "non-present": 0,
            "negative": 1,
            "positive": 2
        }
        for r in result:
            if r['label'] in ["non-present", "negative", "positive"]:
                pred = mapping[r['label']]

        all_predictions.append(pred)

    # Evaluate the model
    print('Evaluating the model...')
    y_true = test_data['label']
    print("Evaluation metrics for the model:")
    evaluate_model_per_aspect(y_true, all_predictions, model_name=model_name.split("/")[-1] + "_model_zero_shot",aspect=aspect)