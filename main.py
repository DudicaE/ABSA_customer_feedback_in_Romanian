from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
import matplotlib.pyplot as plt
from functions.prediction_class import predict_reviews
import plotly.express as px
import plotly.graph_objects as go
app = Flask(__name__)
df_aspect = pd.read_csv('data/golden_standard/model_one_dataset.csv',sep=';', index_col=0)
df_polarity = pd.read_csv('data/golden_standard/model_two_dataset.csv',sep=';', index_col=0)
UPLOAD_FOLDER = 'dashboard_data'
OUTPUT_FOLDER = 'predicted_output'
ASPECT_COLUMNS = ["design", "functionalitate", "livrare_ambalaj", "calitate_pret", "satisfactie_generala"]
POLARITY_MAPPING = {0:'Absent',1:'Negative',2:'Positive'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html', page_name='Homepage')

@app.route('/dashboard/<filename>')
def dashboard(filename):
    filepath = os.path.join(OUTPUT_FOLDER, filename)

    df = pd.read_csv(filepath, sep=";")
    dataset_fig = go.Figure(data=[go.Table(
    header=dict(values=list(df.columns),
                fill_color='#4AA1FF',
                align='left'),
    cells=dict(values=[df.content, df.design, df.functionalitate,df.livrare_ambalaj,df.calitate_pret,df.satisfactie_generala],
               fill_color="#B3D5FF",
               align='left'))
    ])
    dataset_fig.update_layout(width=1100,height=550)
    dataset_fig = dataset_fig.to_html()

    # Aspect df create for the Aspect barchart graph
    aspect_df =(df[ASPECT_COLUMNS].melt(var_name='Aspect',value_name='Aspect_presence'))
    aspect_df['Aspect_presence'] = aspect_df['Aspect_presence'].apply(lambda x: 'Absent' if x == 0 else 'Present')

    # Polarity df create for the Polarity stacked barchart graph
    polarity_df = (df[ASPECT_COLUMNS].melt(var_name="Aspect", value_name="Polarity"))
    polarity_df = polarity_df[polarity_df['Polarity'] != 0]
    polarity_df["Polarity"] = polarity_df["Polarity"].map(POLARITY_MAPPING)

    # Variables to automatically display on the report 
    total_values = len(df[ASPECT_COLUMNS].values.flatten())

    present_count = (df[ASPECT_COLUMNS] != 0).sum().sum()
    absent_count = (df[ASPECT_COLUMNS] == 0).sum().sum()

    present_percentage = round((present_count / total_values) * 100, 1)
    absent_percentage = round((absent_count / total_values) * 100, 1)
    
    polarity_values = df[ASPECT_COLUMNS].values.flatten()
    polarity_values = polarity_values[polarity_values != 0]

    positive_count = (polarity_values == 2).sum()
    negative_count = (polarity_values == 1).sum()
    total_sentiment_polarity = positive_count + negative_count

    positive_percentage = round((positive_count/total_sentiment_polarity)*100,1)
    negative_percentage = round((negative_count/total_sentiment_polarity)*100,1)

    if positive_count >= negative_count:
        overall_sentiment = "POSITIVE"
        overall_sentiment_percentage = positive_percentage

    else:
        overall_sentiment = "NEGATIVE"
        overall_sentiment_percentage = negative_percentage

    fig = px.histogram(
        aspect_df,
        x="Aspect",
        color="Aspect_presence",
        barmode="group",
        color_discrete_map={
                "Absent": "#F5E9D8",
                "Present": "#2FA4D7"
                },
        title="Distribution of the Aspects detected in the reviews"
    )
    fig.update_layout(width=1100,height=550)
    graph_html = fig.to_html()
    
    fig1 = px.histogram(
        polarity_df,
        y="Aspect",
        color="Polarity",
        barmode="group",
        color_discrete_map={
                "Negative": "#FF4A4A",
                "Positive": "#2AD100",
                },
        title="Sentiment distribution of the present aspects detected in the reviews"
    )
    fig1.update_layout(width=1100,height=550)
    graph_html1 = fig1.to_html()

    return render_template(
        'dashboard.html',
        page_name='Dashboard',
        filename=filename,
        alt_tabel=dataset_fig,
        graph1=graph_html,
        graph2=graph_html1,
        present_count=present_count,
        absent_count=absent_count,
        present_percentage=present_percentage,
        absent_percentage=absent_percentage,
        positive_count=positive_count,
        negative_count=negative_count,
        positive_percentage=positive_percentage,
        negative_percentage=negative_percentage,
        overall_sentiment = overall_sentiment,
        overall_sentiment_percentage = overall_sentiment_percentage
    )

@app.route('/upload-csv', methods=['POST'])
def upload_csv():
    if "csv_file" not in request.files:
        return "Error: There was no .csv file uploaded"

    file = request.files["csv_file"]

    filename = file.filename

    filepath = os.path.join("dashboard_data", filename)

    file.save(filepath)
    number_files = len(os.listdir("./predicted_output"))
    output_file=f"./predicted_output/{filename.replace('.csv', str(number_files) +'.csv')}"
    # Reading the reviews from the saved .csv file
    predict_reviews(
        file_name=filepath,
        output_file=output_file,
        aspect_model_path="./results/readerbench/robert-base_fine_tuning/checkpoint-750", 
        polarity_model_path="./results/readerbench/robert-base_", 
        label_names=["content", "design", "functionalitate", "livrare_ambalaj", "calitate_pret", "satisfactie_generala"]
        )

    return {"redirect_url": url_for("dashboard", filename=f"{filename.replace('.csv', str(number_files) +'.csv')}")
}


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)