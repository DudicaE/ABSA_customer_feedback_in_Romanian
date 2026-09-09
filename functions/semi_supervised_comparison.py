#File which compares the semi-supervised trained model with the golden standard model over 1000 reviews manually annotated dataset.

# Using the prediction class function to predict the aspect and polarity of the 1000 reviews and save the results in a csv file for both models. Then the results of the 4000 reviews dataset with a semi-supervised trained model will be compared with the results of the golden standard model. This is done in order to asses the accuracy of the model trained on semi-supervised dataset. The accuracy will be evaluated using the F1 score, precision and recall metrics for each aspect and polarity of the reviews. The results will be saved in a csv file for further analysis.


import os

from functions.prediction_class import predict_reviews
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

aspect = ["content","design", "functionalitate", "livrare_ambalaj", "calitate_pret", "satisfactie_generala"]

def generate_predictions():
    """Function which generates the predictions for the 1000 reviews dataset using both the semi-supervised trained model and the golden standard model. The predictions are saved in separate csv files for further analysis."""
    # Predicting the aspect and polarity of the 1000 reviews using the golden standard model
    predict_reviews(
        file_name="data/test_data/model_two_dataset.csv",
        label_names=aspect,
        output_file="data/test_data/predicted_1000_reviews_semi_supervised.csv",
        aspect_model_path="./results/readerbench/robert-base_fine_tuning/checkpoint-400",
        polarity_model_path="./results/readerbench/Robert-base_",
        separator="|"
    )
    
    # Predicting the aspect and polarity of the 1000 reviews using the semi-supervised trained model
    predict_reviews(
        file_name="data/test_data/model_two_dataset.csv",
        label_names=aspect,
        output_file="data/test_data/predicted_1000_reviews_golden_standard.csv",
        aspect_model_path="./results/semi_supervised/robert-base_fine_tuning/checkpoint-400",
        polarity_model_path="./results/semi_supervised/Robert-base_",
        separator="|"
    )


