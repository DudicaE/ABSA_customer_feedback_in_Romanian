from posixpath import sep

import pandas as pd 
from sklearn.model_selection import train_test_split
from datasets import Dataset

TARGET_FILE_PATH = "data/raw_xlsx/data_reviews_annotated.xlsx"
OUTPUT_FILE_PATH = "data/raw_models_data.csv"
LABEL_COL_NAME = ["content","design", "functionalitate", "livrare_ambalaj", "calitate_pret", "satisfactie_generala"]
DATA_PATH = 'data/'

# Function for removal of special characters out of the dataset for training
def removal_special_characters(df:pd.DataFrame,column:str):
    """For removing all the special letters with the regular ones from Romanian language and returning the Dataframe. Example: 'ă / â'	 charactres will be replaced by 'a'"""
    special_character_list = ["ă", "â", "î", "ș", "ț"]
    normalized_character_list = ["a", "a", "i", "s", "t"]
    for i in range(len(special_character_list)):
        df[column] = df[column].str.lower()
        df[column] = df[column].str.replace(f'{special_character_list[i]}', f'{normalized_character_list[i]}')
    return df

# Function for creation of the first model dataset which evaluates the existence of the aspects in the reviews in a binary format. The csv file is checked if it exists and if it's not then is created from data/annotated_reviews_LaRoSeDa.csv by replacing the labels from 1,2,0 to 1,0 for the existence of the aspects in the reviews. The file is saved in the data folder as model_one_dataset.csv
def first_model_dataset_creation(input_file: str = "data/annotated_reviews_LaRoSeDa.csv", output_file: str = "data/golden_standard/model_one_dataset.csv", separator:str = ";"):
    """Function which checks and creates the dataset for the first model where the existence of the aspects of the review are mapped 0: non existent, 1: existent. In the end the file is save in the data directory"""

    encode_map = {1: 1, 2: 1, 0: 0}

    try:
        with open(output_file,"r") as f :
            print("Dataset file exists.")
    except FileNotFoundError:
        data = pd.read_csv(input_file,sep=separator)
        data = removal_special_characters(df=data,column='content')
        model_one_dataset = data[LABEL_COL_NAME].copy()
        model_one_dataset[LABEL_COL_NAME[1:]] = model_one_dataset[LABEL_COL_NAME[1:]].replace(encode_map)
        model_one_dataset.to_csv(output_file, sep=separator)

def second_model_dataset_creation(input_file: str = "data/annotated_reviews_LaRoSeDa.csv", output_file: str = "data/golden_standard/model_two_dataset.csv", separator:str = ";"):
    """Function which checks and creates the dataset for the second model where the polarity existence of the aspects of the review are mapped 0: non existent, 1:negative, 2:positive. In the end the file is save in the data directory"""


    try:
        with open(output_file,"r") as f :
            print("Dataset file exists.")
    except FileNotFoundError:
        data = pd.read_csv(input_file,sep=separator)
        data = removal_special_characters(df=data,column='content')
        model_two_dataset = data[LABEL_COL_NAME].copy()
        model_two_dataset.to_csv(output_file, sep=separator)

#Function for splitting the model two dataset from a multi-label multi-output dataset to a single-label multi-output dataset for the evaluation of the model on each aspect separately
def second_model_dataset_spliting(input_file: str = "data/golden_standard/model_two_dataset.csv", output_file: str = "data/golden_standard/model_two_{aspect}_dataset.csv", separator:str = ";"):
    data = pd.read_csv(input_file,sep=separator)
    print("Dataset for model two has been loaded successfully.")
    for aspect in LABEL_COL_NAME[1:]:
        print(f"Processing aspect: {aspect}")
        aspect_dataset = data[["content", aspect]].copy()
        aspect_dataset.to_csv(output_file.format(aspect=aspect), sep=separator)



# Loading data for the models
def load_data(file_name:str, seed:int, test_size:float):
    """ Returning the training dataset and the testing datasets in a dictionary format plus the name of the columns for labels."""

    data = pd.read_csv(DATA_PATH + file_name, sep=';')

    X = data['content'].tolist()
    y = data[LABEL_COL_NAME[1:]].astype(float).values.tolist()  # Convert to float for multi-label classification

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = seed, shuffle = True)

    train_ds = Dataset.from_dict({"content": X_train, "label": y_train})
    test_ds   = Dataset.from_dict({"content": X_test,   "label": y_test})
    return train_ds, test_ds, LABEL_COL_NAME[1:]

def load_data_single_label(file_name:str, seed:int, test_size:float, aspect:str):
    """ Returning the training dataset and the testing datasets in a dictionary format plus the name of the columns for labels for the single label multi output dataset for the second model."""

    data = pd.read_csv(DATA_PATH + file_name, sep=';')

    X = data['content'].tolist()
    y = data[aspect].astype(int).values.tolist()  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = seed, shuffle = True)

    train_ds = Dataset.from_dict({"content": X_train, "label": y_train})
    test_ds   = Dataset.from_dict({"content": X_test,   "label": y_test})
    return train_ds, test_ds, aspect