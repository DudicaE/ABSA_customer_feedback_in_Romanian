import numpy as np
import pandas as pd 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from functions.data_cleaning import load_data
import time
from functions.evaluation import evaluate_model

def prompt_few_shot(language:str,test_data:pd.DataFrame,i:int,example_labels:list,review_example:str)->str:
    """Function which creates the prompt for few-shot classification. The function will check for which model the prompt is created and which language the prompt is in and will create the prompt accordingly. The function returns the prompt as a string."""

    if language.lower() == "en":
        prompt = [{
            "role": "system", "content": "You are a multi-label text classifier for Romanian costumer reviews. You will be given a review and you have to classify it in 5 aspects: design_calitate_produs, functionalitate_performanta_produs, livrare_ambalaj, pret_valoare, satisfactie_generala. You have to answer only with a list of 5 numbers such [1,0,0,1,0] which represent the presence or absence of the aspects in the order mentioned before."
            },
            {"role": "user",
                "content": f"Example labels: {example_labels} and for these reviews{review_example}\nReview: {test_data['content'][i]}\nAnswer only with a list of 5 numbers such [1,0,0,1,0] which represent the presence or absence of the aspects in the order mentioned before."
            }
]
    elif language.lower() == "ro":
        prompt = [{
            "role": "system",
            "content": "Ești un clasificator de text multi-label pentru recenziile clienților români. Vei primi o recenzie și trebuie să o clasifici în 5 aspecte: design_calitate_produs, functionalitate_performanta_produs, livrare_ambalaj, pret_valoare, satisfactie_generala. Trebuie să răspunzi doar cu o listă de 5 numere, cum ar fi [1,0,0,1,0] care reprezintă prezența sau absența aspectelor în ordinea menționată anterior."},
            {"role": "user",
                "content": f"De exemplu etichete: {example_labels} pentru urmatoarele review-uri: {review_example}\nRecenzie: {test_data['content'][i]}\nRăspunde doar cu o listă de 5 numere, cum ar fi [1,0,0,1,0] care reprezintă prezența sau absența aspectelor în ordinea menționată anterior."}
]
    return prompt


def classification_n_shot(pipe,message:list,tokenizer:AutoTokenizer)->list:
    """Function which performs multi-label classification using the model. The function takes as input the pipeline for the model, the message template and the review to be classified. The function returns the predicted labels for the review as a list of integers."""
    input_text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(input_text, return_tensors="pt").to(pipe.device)
    outputs = pipe.model.generate(**inputs, max_new_tokens=32768, do_sample=False)
    predicted_labels = tokenizer.decode(outputs, skip_special_tokens=True)
    index = predicted_labels[0].find("</think>") + len("</think>")  # Find the index of the end of the thinking process
    print(f"Index of </think>: {index}")
    predicted_labels = predicted_labels[0]
    predicted_labels = predicted_labels[index:].strip("\n")  # Get the text after the thinking process
    print(f"Raw model output: {predicted_labels}")
    if predicted_labels.startswith("[" ) and predicted_labels.endswith("]"):
        predicted_labels = predicted_labels[1:-1].split(",")
        # print(f"Predicted labels after removing brackets and splitting: {predicted_labels}")
        predicted_labels = [int(label.strip()) for label in predicted_labels]
        # print(f"Predicted labels after converting to integers: {predicted_labels}")
    else:
        predicted_labels = [int(label.strip()) for label in predicted_labels.split(",")]
        # print(f"Predicted labels after splitting without removing brackets: {predicted_labels}")
    print(f"Predicted labels: {predicted_labels}")
    return predicted_labels

def few_shot_classification(file_name:str,model_name:str,model_id:int=1,language:str="en"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the model and tokenizer 
    train_data, test_data, label_columns = load_data(file_name=file_name, seed=5, test_size=0.2)
    tokenizer  = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype="auto").to(device)
    print('Model and tokenizer loaded')

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=200,
        do_sample=False,
    )

    # Few-shot classification
    print('Starting few-shot classification...')

    predicted_labels = []
    # Few-shot classification
    for i in range(len(test_data)):
        start_time = time.time()
        #5 random examples from the training dataset to be used as examples in the few-shot prompt saved in text example and labels example variables
        text_example = train_data['content'][:5]
        example_labels = train_data['label'][:5]
        prompt = prompt_few_shot(language=language, test_data=test_data, i=i, example_labels=example_labels, review_example=text_example)
        true_labels_list = test_data['label'][i]

        predicted_label_list = classification_n_shot(pipe=pipe, message=prompt, tokenizer=tokenizer)
        predicted_labels.append(predicted_label_list)
        end_time = time.time()
        print(f"Processed review {i+1}/{len(test_data)} in {end_time - start_time:.2f} seconds.")
        print(f"True labels: {true_labels_list}, Predicted labels: {predicted_label_list}\n")

    # Evaluate the model
    print('Evaluating the model...')
    y_true = test_data['label']
    print("Evaluation metrics for the model:")
    evaluate_model(y_true, predicted_labels, model_name=model_name.split("/")[-1] + "_" + language + "_model_" + str(model_id) + "_few_shot")
