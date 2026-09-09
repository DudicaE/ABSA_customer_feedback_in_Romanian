import pandas as pd 
import numpy as np 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import os
from functions.data_cleaning import LABEL_COL_NAME
import matplotlib.pyplot as plt

# Function for evaluation of the models metrics

def evaluate_model(y_true, y_pred, model_name:str):
    """Function which takes as input the true labels and the predicted labels and returns and saves the classification report and the confusion matrix for each label in a file in the results directory."""
    
    #Ensuring that the predicted labels are in the same format as the true labels
    y_pred = [eval(predicted) if isinstance(predicted, str) else predicted for predicted in y_pred]
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    # if not os.path.exists("results"):
    #     os.makedirs("results")
    if not os.path.exists(f"results/{model_name}"):
        os.makedirs(f"results/{model_name}")
    report_df.to_csv(f"results/{model_name}/classification_report_{model_name}.csv", index=True)
    print(f"\nClassification report for {model_name} saved to results/{model_name}/classification_report_{model_name}.csv"+"\n"+str(report))

    # Save the confusion matrix images for each label
    for i, label in enumerate(LABEL_COL_NAME[1:]):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])

        cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
        cm_df.to_csv(f"results/{model_name}/confusion_matrix_{model_name}_{label}.csv", index=True)

        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        cm_display.plot(cmap='Blues')
    
        plt.title(f"Confusion Matrix for {model_name.split('/')[-1]}-{label}")
        cm_display.figure_.savefig(f"results/{model_name}/confusion_matrix_{model_name}_{label}.png")

        print(f"\nConfusion matrix for {model_name} and label {label} saved to results/{model_name}/confusion_matrix_{model_name}_{label}.csv")


def evaluate_model_per_aspect(y_true, y_pred, model_name:str, aspect:str):
    """Function which takes as input the true labels and the predicted labels for a specific aspect and returns and saves the classification report and the confusion matrix for that aspect in a file in the results directory."""
    
    #Ensuring that the predicted labels are in the same format as the true labels
    y_pred = [eval(predicted) if isinstance(predicted, str) else predicted for predicted in y_pred]
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    os.makedirs(f"results/{model_name}_polarity", exist_ok=True)
    report_df.to_csv(f"results/{model_name}_polarity/classification_report_{model_name}_{aspect}.csv", index=True)
    print(f"Classification report for {model_name} and aspect {aspect} saved to results/{model_name}y/classification_report_{model_name}_{aspect}.csv"+"\n"+str(report))

    # Save the confusion matrix image for the aspect polarity classification multi-output (0,1,2)
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1', 'Actual 2'], columns=['Predicted 0', 'Predicted 1', 'Predicted 2'])
    cm_df.to_csv(f"results/{model_name}_polarity/confusion_matrix_{model_name}_{aspect}.csv", index=True)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2])
    cm_display.plot(cmap='Blues')
    plt.title(f"Confusion Matrix for{model_name.split('/')[-1]}-{aspect}")
    cm_display.figure_.savefig(f"results/{model_name}_polarity/confusion_matrix_{model_name}_{aspect}.png")
    print(f"\nConfusion matrix for {model_name} and aspect {aspect} saved to results/{model_name}_polarity/confusion_matrix_{model_name}_{aspect}.csv")



def processing_files_predictions_for_evaluation(file_name_pred:str,file_name_true:str,separator1:str="|",separator2:str="|", aspects:list=["design", "functionalitate", "livrare_ambalaj", "calitate_pret", "satisfactie_generala"]):
    """Function which takes as input the file names of the predicted labels and the true labels and returns the true labels and the predicted labels in a format suitable for evaluation."""
    df_pred = pd.read_csv(file_name_pred, sep=separator1)
    df_true = pd.read_csv(file_name_true, sep=separator2)
    y_pred = df_pred[aspects].values
    y_true = df_true[aspects].values
    return y_true, y_pred
