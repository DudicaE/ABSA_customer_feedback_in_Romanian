import numpy as np
import pandas as pd 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from functions.data_cleaning import load_data_single_label
import time
from functions.evaluation import evaluate_model_per_aspect


def prompt_few_shot(language:str,aspect:str,example_labels:list,review_example:list,test_data:pd.DataFrame,i:int)->str:
    """Function which creates the prompt for few-shot classification. The function will check for which model the prompt is created and which language the prompt is in and will create the prompt accordingly. The function returns the prompt as a string."""

    if language.lower() == "en":
        prompt = [{"role": "system", 
                        "content": f"You are a text classifier for Romanian costumer reviews. You will be given a review and you have to classify it based on the aspect: {aspect}. You have to answer only with a one of the following integers which represent the polarity of the aspect as follows: 0: non existent, 1:negative, 2: positive."},
                        {"role": "user", 
                        "content": f"Example labels: {example_labels} for these reviews: {review_example}\nReview: {test_data['content'][i]}\nAnswer only with an integer number represented as follows: 0: non existent, 1:negative, 2:neutral, 3: positive. The number will represent the polarity of the aspect mentioned before."}
]
    elif language.lower() == "ro":
        prompt = [{"role": "system",
                        "content": f"Esti un clasificator de text pentru recenziile clienților români. Vei primi o recenzie și trebuie să o clasifici în funcție de aspectul: {aspect}. Trebuie să răspunzi doar cu unul dintre următoarele numere întregi care reprezintă polaritatea aspectului după cum urmează: 0: inexistent, 1: negativ, 2: pozitiv."},
                        {"role": "user",
                        "content": f"Example labels: {example_labels} for these reviews: {review_example}\nRecenzie: {test_data['content'][i]}\nRăspunde doar cu un număr întreg, cum ar fi 0, 1, 2 sau 3, care reprezintă polaritatea aspectului menționat anterior. Polaritatea este reprezentată astfel: 0: inexistent, 1: negativ, 2: pozitiv."}
]
    return prompt
def classification_n_shot(pipe,message:list,tokenizer:AutoTokenizer)->list:
    """Function which performs few-shot classification per aspect instead of multi-label classification.
    The function takes as input the pipeline for the model, the message template and the review to be classified. The function returns the predicted label for the aspect as an integer."""
    input_text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(input_text, return_tensors="pt").to(pipe.device)
    outputs = pipe.model.generate(**inputs, max_new_tokens=32768, do_sample=False)
    predicted_labels = tokenizer.decode(outputs, skip_special_tokens=True)
    index = predicted_labels[0].find("</think>") + len("</think>")  # Find the index of the end of the thinking process
    print(f"Index of </think>: {index}")
    predicted_labels = predicted_labels[0]
    predicted_labels = predicted_labels[index:].strip("\n")  # Get the text after the thinking process
    print(f"Raw model output: {predicted_labels}")
    #Check if the predicted label is one of the expected integers (0,1,2) and if so convert it to an int,otherwise check if it is a string like [0,1,2] and extract the integer from it if it's the same integer across the list else return 0 as default value for non existent aspect
    if predicted_labels.startswith("[" ) and predicted_labels.endswith("]"):
        predicted_labels = predicted_labels[1:-1].split(",")
        predicted_labels = [int(label.strip()) for label in predicted_labels]
        if len(set(predicted_labels)) == 1:
            predicted_labels = predicted_labels[0]
        else:
            predicted_labels = 0
    elif predicted_labels in ["0", "1", "2"]:
        predicted_labels = int(predicted_labels)
    else:
        predicted_labels = 0
    print(f"Predicted labels: {predicted_labels}")
    return predicted_labels

def few_shot_classification_two(file_name:str,model_name:str,aspect:str,model_id:int=2,language:str="en"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the model and tokenizer 
    train_data, test_data, label_columns = load_data_single_label(file_name=file_name,aspect=aspect, seed=5, test_size=0.2)
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

    for i in range(len(test_data)):
        start_time = time.time()
        # 5 random examples from the training dataset to be used as examples in the few-shot prompt
        text_example = train_data['content'][:5]
        example_labels = train_data['label'][:5]
        prompt = prompt_few_shot(language=language, aspect=aspect, example_labels=example_labels, review_example=text_example, test_data=test_data, i=i)
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
    evaluate_model_per_aspect(y_true, predicted_labels, model_name=model_name.split("/")[-1] + "_" + language + "_model_" + str(model_id) + "_few_shot", aspect=aspect)
