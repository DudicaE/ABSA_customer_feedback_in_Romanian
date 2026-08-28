from functions.preprocessing_data import read_json_and_convert_to_dataframe, generate_wordcloud, combine_dataframes, star_rating_distribution_viz, star_rating_review_length_correlation
import pandas as pd 
file = 'data/negative_reviews.json'
file1 = 'data/positive_reviews.json'

if __name__ == "__main__":

    # generate_wordcloud(file,None,save_path='images/wordclouds/negative_reviews.png',viz_name="LaRoSeDa Negative Reviews Word Cloud Visualization")
    # generate_wordcloud(file1,None,save_path='images/wordclouds/positive_reviews.png',viz_name="LaRoSeDa Positive Reviews Word Cloud Visualization")

    # Dataframes combination and saving to csv file
    # df = read_json_and_convert_to_dataframe(file)
    # df1 = read_json_and_convert_to_dataframe(file1)

    # combined_df = combine_dataframes(df, df1, save_path='data/combined_reviews.csv')

    # data = pd.read_csv('data/combined_reviews.csv')
    # print(data.head())
    # print(data.info())

    # star_rating_distribution_viz(data, save_path='images/dataset/star_rating_distribution.png')
    # star_rating_review_length_correlation(data, save_path='images/dataset/star_rating_review_length_correlation.png')
    
    data = pd.read_csv('data/combined_reviews.csv')
    # Selecting  2000 reviews, 1000 positive and 1000 negative, out of the 1000 positive  25% will be from 4 stars and 75% from 5 stars, as from 1000 negative reviews 25% will be from 2 stars and 75% from 1 star for the ABSA annotation task and saving them as a new csv file and removing the original combined_reviews.csv and generate a new combined_reviews.csv file with the remaining reviews for future use in other tasks.
    positive_reviews_4_stars = data[data['starRating'] == 4].sample(n=250, random_state=1)
    positive_reviews_5_stars = data[data['starRating'] == 5].sample(n=750, random_state=1)
    negative_reviews_1_star = data[data['starRating'] == 1].sample(n=750, random_state=1)
    negative_reviews_2_stars = data[data['starRating'] == 2].sample(n=250, random_state=1)
    selected_reviews = pd.concat([positive_reviews_4_stars, positive_reviews_5_stars, negative_reviews_1_star, negative_reviews_2_stars], ignore_index=True)
    selected_reviews.to_csv('data/annotated_reviews.csv', index=False)
    remaining_reviews = data.drop(selected_reviews.index)
    remaining_reviews.to_csv('data/dataset_reviews.csv', index=False)




