from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd 
import numpy as np
from functions.data_cleaning import removal_special_characters

# Class which will be used for prediction of the labels for the reviews. It will load the model and tokenizer and will be used for prediction of the labels for the reviews.

class PredictionModel:
    def __init__(self,aspect_model_path:str, polarity_model_path:str, label_names:list):
        # Load the models and the tokenizers for the aspect and polarity prediction
        self.aspect_tokenizer = AutoTokenizer.from_pretrained(aspect_model_path)
        self.aspect_model = AutoModelForSequenceClassification.from_pretrained(aspect_model_path)

        # Completing the paths accrodingly to the aspect
        for aspect in label_names[1:]:
            model_path = polarity_model_path + aspect + "/checkpoint-200"
            setattr(self, f"{aspect}_tokenizer", AutoTokenizer.from_pretrained(model_path))
            setattr(self, f"{aspect}_model", AutoModelForSequenceClassification.from_pretrained(model_path))

        self.label_names = label_names
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.aspect_model.to(self.device)
        for aspect in label_names[1:]:
            getattr(self, f"{aspect}_model").to(self.device)
        
    def predict_aspect(self, review:str):
        # Predict the aspect presence for the review under a list format 
        aspect_input = self.aspect_tokenizer(review, return_tensors="pt", truncation=True, padding=True).to(self.device)

        with torch.no_grad():
            aspect_outputs = self.aspect_model(**aspect_input)
        aspect_logits = torch.sigmoid(aspect_outputs.logits).cpu().numpy()[0]
        aspect_predictions = (aspect_logits > 0.5).astype(int)

        return  aspect_predictions.tolist()


    def predict_polarity(self, review:str, aspect:str):
        # Predict the polarity for the review and aspect under a list format 
        polarity_tokenizer = getattr(self, f"{aspect}_tokenizer")
        polarity_model = getattr(self, f"{aspect}_model")
        polarity_input = polarity_tokenizer(review, return_tensors="pt", truncation=True, padding=True).to(self.device)

        with torch.no_grad():
            polarity_outputs = polarity_model(**polarity_input)
        polarity_logits = torch.softmax(polarity_outputs.logits, dim=-1).cpu().numpy()[0]
        polarity_prediction = np.argmax(polarity_logits)

        return polarity_prediction    


def predict_reviews(file_name:str, output_file:str,aspect_model_path:str, polarity_model_path:str, label_names:list, separator:str=";"):
    df = pd.read_csv(file_name,sep=separator)
    df = removal_special_characters(df,'content')

    aspect_predictions = []
    polarity_predictions = []
    prediction_model = PredictionModel(
        aspect_model_path=aspect_model_path,
        polarity_model_path=polarity_model_path,
        label_names=label_names
    )

    for review in df["content"]:
        aspect_pred = prediction_model.predict_aspect(review)
        aspect_predictions.append(aspect_pred)
        local_polarity_predict_list = []
        if aspect_pred[0] == 1:
            polarity_pred = prediction_model.predict_polarity(review, "design")
            local_polarity_predict_list.append(polarity_pred)
        else:
            local_polarity_predict_list.append(0)

        if aspect_pred[1] == 1:
            polarity_pred = prediction_model.predict_polarity(review, "functionalitate")
            local_polarity_predict_list.append(polarity_pred)
        else:
            local_polarity_predict_list.append(0)

        if aspect_pred[2] == 1:
            polarity_pred = prediction_model.predict_polarity(review, "livrare_ambalaj")
            local_polarity_predict_list.append(polarity_pred)
        else:
            local_polarity_predict_list.append(0)

        if aspect_pred[3] == 1:
            polarity_pred = prediction_model.predict_polarity(review, "calitate_pret")
            local_polarity_predict_list.append(polarity_pred)
        else:
            local_polarity_predict_list.append(0)

        if aspect_pred[4] == 1:
            polarity_pred = prediction_model.predict_polarity(review, "satisfactie_generala")
            local_polarity_predict_list.append(polarity_pred)
        else:
            local_polarity_predict_list.append(0)
        polarity_predictions.append(local_polarity_predict_list)

    # Converting the prediction lists in to arrays to multiply the aspecs predictions with the polarity predictions and to have the final predictions for the aspects and the polarities
    aspect_predictions = np.array(aspect_predictions)
    polarity_predictions = np.array(polarity_predictions)
    final_predictions = aspect_predictions * polarity_predictions
    final_df = pd.DataFrame(final_predictions, columns=["design", "functionalitate", "livrare_ambalaj", "calitate_pret", "satisfactie_generala"])
    final_df.insert(0, "content", df["content"])
    print(output_file)
    final_df.to_csv(output_file,sep=separator, index=False)
    return final_df