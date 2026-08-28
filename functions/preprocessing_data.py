import json
import pandas as pd 
from wordcloud import WordCloud 
import matplotlib.pyplot as plt
import spacy 

def removal_special_characters(df:pd.DataFrame,column:str):
    """ This function removes all the special letters from the Romanian language text and replaces them with the regular ones. For example, 'ă / â' characters will be replaced by 'a'.
    Args:        df (pd.DataFrame): The input DataFrame containing the text data.
        column (str): The name of the column in the DataFrame that contains the text to be processed.
    Returns:        pd.DataFrame: The DataFrame with the special characters removed and replaced.
    """

    df[column] = df[column].str.replace("ă", "a").replace("â", "a").replace("î", "i").replace("ș", "s").replace("ț", "t")
    return df


def read_json_and_convert_to_dataframe(file_path):
    """"
    This functions uses the file path to read the json file and convert it to a pandas DataFrame.
    Args:
        file_path (str): The path to the json file.
    Returns:
        pd.DataFrame: The converted DataFrame.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    df = pd.json_normalize(data['reviews'])
    return df

def filter_tokens_and_lemmatize(dataframe: pd.DataFrame, column: str):
    """This function takes a text as input, processes it using Spacy to filter out stop words, punctuation, spaces, and numbers, and returns a list of lemmatized tokens.
    Args:
        dataframe (pd.DataFrame): The input DataFrame containing the text data.
        column (str): The name of the column in the DataFrame that contains the text to be processed.
    Returns:
        list: A list of lemmatized tokens.
    """
    nlp = spacy.load('ro_core_news_sm')
    nlp.max_length = 2479161
    doc = nlp(''.join(dataframe[column].astype(str).tolist()))
    filtered_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space and not token.like_num]
    return filtered_tokens

def generate_wordcloud(file,tokens,save_path=None,viz_name="LaRoSeDa Word Cloud Visualization"):
    """This function generates a word cloud visualization from a list of tokens.
    Args:
            file (str): The path to the json file containing the text data.
            tokens (list): A list of tokens to be included in the word cloud. If None, the function will generate a default word cloud using the original text data.
            save_path (str, optional): The path where the generated word cloud image will be saved. If None, the image will not be saved.
    Returns:
        None: Displays the generated word cloud.
    """
    if tokens is None or len(tokens) == 0:
        print("Since no tokens were provided the default word cloud will be generated using the original text data.")
        df = read_json_and_convert_to_dataframe(file)
        df = removal_special_characters(df, 'content')
        tokens = filter_tokens_and_lemmatize(df, 'content')
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(tokens))

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  
    plt.title(viz_name)
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
def combine_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, save_path: str = None) -> pd.DataFrame:
        """This function combines two DataFrames into one.
        Args:
            df1 (pd.DataFrame): The first DataFrame to be combined.
            df2 (pd.DataFrame): The second DataFrame to be combined.
        Returns:
            pd.DataFrame: The combined DataFrame.
        """
        combined_df = pd.concat([df1, df2], ignore_index=True)
        if save_path:
            combined_df.to_csv(save_path, index=False)
        print(f"Combined DataFrame has been saved to {save_path}")
        return combined_df

def star_rating_distribution_viz(df: pd.DataFrame, save_path: str = None):
    """This function generates a bar chart visualization of the distribution of star ratings in the combined reviews dataset.
    Args:
        df (pd.DataFrame): The DataFrame containing the star rating data.
        save_path (str, optional): The path where the generated bar chart will be saved. If None, the chart will not be saved.
    Returns:
        None: Displays the generated bar chart.
    """

    star_counts = df['starRating'].value_counts().sort_index()
    print(star_counts)
    plt.figure(figsize=(8, 5))
    plt.bar(star_counts.index.astype(str), star_counts.values, color='skyblue')
    plt.xlabel('Star Rating')
    plt.ylabel('Number of Reviews')
    plt.title('Distribution of Star Ratings')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def star_rating_review_length_correlation(df: pd.DataFrame, save_path: str = None):
    """This function generates a scatter plot visualization of the correlation between star ratings and review lengths in the combined reviews dataset.
    Args:
        df (pd.DataFrame): The DataFrame containing the star rating and review length data.
        save_path (str, optional): The path where the generated scatter plot will be saved. If None, the plot will not be saved.

    Returns:
        None: Displays the generated scatter plot.
    """
    df['starRating'] = df['starRating'].astype(str)
    df['review_length'] = df['content'].apply(lambda x: round(len(str(x))))
    print(df[['starRating', 'review_length']].head())
    plt.figure(figsize=(8, 5))
    plt.scatter(df['starRating'], df['review_length'],)
    plt.xlabel('Star Rating')
    plt.ylabel('Review Length')
    plt.title('Correlation between Star Ratings and Review Lengths')
    if save_path:
        plt.savefig(save_path)
    plt.show()