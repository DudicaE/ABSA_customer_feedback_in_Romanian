import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

aspect_columns = ["design", "functionalitate", "livrare_ambalaj", "calitate_pret", "satisfactie_generala"]
polarity_color_mapping = {0:'blue',1:'red',2:'green'}
df_aspect = pd.read_csv('data/model_one_dataset.csv',sep=';')
df_polarity = pd.read_csv('data/model_two_dataset.csv',sep=';')

def data_head(df, n=5):
    return df.head(n)


def data_describe(df):
    return df.describe()

def data_types(df):
    return df.dtypes

data_head = data_head(df_aspect)
data_describe = data_describe(df_aspect)
data_types = data_types(df_aspect)
print("Data Head:")
print(data_head)
print("\nData Describe:")
print(data_describe)
print("\nData Types:")
print(data_types)


# Visualize the distribution distribution of the aspects presence in the dataset

# Plot the distribution of each aspect and show in legends the numbers of reviews for each aspect
def plot_aspect_distribution(df, aspect_columns):
    for aspect in aspect_columns:
        plt.figure(figsize=(6, 4))
        df_aspect[aspect].value_counts().plot(kind='bar')
        plt.title(f'Distribution of {aspect}')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        if df_aspect[aspect].value_counts()[0] > df_aspect[aspect].value_counts()[1]:
            plt.legend([f'Class 0 : {df_aspect[aspect].value_counts()[0]} Class 1 : {df_aspect[aspect].value_counts()[1]}'])
        else:
            plt.legend([f'Class 0 : {df_aspect[aspect].value_counts()[1]} Class 1 : {df_aspect[aspect].value_counts()[0]}'])
        plt.savefig(f'data/images/{aspect}_distribution_model_1.png')
        plt.show()

    
    

# Visualize the polarity distribution in the dataset
def plot_polarity_distribution(df, aspect_columns):
    for aspect in aspect_columns:
        plt.figure(figsize=(6, 4))
        df_polarity[aspect].value_counts().plot(kind='bar')
        plt.title(f'Polarity Distribution of {aspect}')
        plt.xlabel('Polarity')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.savefig(f'data/images/{aspect}_polarity_distribution_model_2.png')
        plt.show()


# Visualization for the correlation between lenght of the review and it's polarity 
def plot_review_length_polarity_correlation(df, aspect_columns):
    for aspect in aspect_columns:
        plt.figure(figsize=(6, 4))
        df_polarity['review_length'] = df_polarity['content'].apply(lambda x: len(str(x).split()))
        df_polarity.boxplot(column='review_length', by=aspect)
        plt.title(f'Review Length and Polarity of {aspect}')
        plt.xlabel('Polarity')
        plt.ylabel('Review Length')
        plt.xticks(rotation=0)
        plt.savefig(f'data/images/{aspect}_review_length_polarity_correlation_model_2.png')
        plt.show()

plot_aspect_distribution(df_aspect, aspect_columns)
plot_polarity_distribution(df_polarity, aspect_columns)
plot_review_length_polarity_correlation(df_polarity, aspect_columns)