import streamlit as st
import pandas as pd
import plotly.express as px
import requests

# Title and introduction
st.title("Restaurant Ratings Analysis")
st.write("""
## Introduction
Restaurant ratings serve as a valuable reference for both consumers and restaurants. Restaurant ratings influence how much money a restaurant makes and help customers choose where to eat.
""")

@st.cache
def load_data():
    try:
        data = pd.read_csv('ML_Data.csv')  # Update this with the path to your data file
        if 'Unnamed: 0' in data.columns:
            data = data.drop('Unnamed: 0', axis=1)
        return data
    except FileNotFoundError:
        st.error("The file was not found. Please ensure the file is in the correct location.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return None

# Load the data
data = load_data()

# Check if data is loaded successfully
if data is not None:
    st.write("Restaurant Ratings Analysis")
    st.dataframe(data.head())  # Display the first few rows of the dataframe
else:
    st.write("No data to display.")

# Visualization: Distribution of Ratings
st.write("## Distribution of Ratings")
rating_counts = data['Rating'].value_counts()
fig = px.bar(rating_counts, x=rating_counts.index, y=rating_counts.values, labels={'x': 'Rating', 'y': 'Count'}, title='Distribution of Ratings')
st.plotly_chart(fig)

# Visualization: Scatter Plot of Ratings vs. Number of Ratings
st.write("## Ratings vs. Number of Ratings ")
fig = px.scatter(data, x='Number of Ratings', y='Rating', title='Ratings vs. Number of Ratings', labels={'Number of Ratings': 'Number of Ratings', 'Rating': 'Rating'})
st.plotly_chart(fig)

# Function to make API requests to FastAPI
def predict_rating_cluster(Number_of_Ratings, Weighted_Rating, Rating_category_encoder):
    payload = {
        "Number_of_Ratings": Number_of_Ratings,
        "Weighted_Rating": Weighted_Rating,
        "Rating_category_encoder": Rating_category_encoder
    }
    response = requests.post("https://use-case-8-project-5.onrender.com", json=payload)
    if response.status_code == 200:
        return response.json()["cluster_labels"]
    else:
        return None

# Sidebar for input
st.sidebar.header("Prediction Input")
Number_of_Ratings = st.sidebar.number_input("Number of Ratings", value=0.0, step=1.0)
Weighted_Rating = st.sidebar.number_input("Weighted Rating", value=0.0)
Rating_category_encoder = st.sidebar.number_input("Rating category encoder", value=0)

# Prediction
if st.sidebar.button("Predict Rating Cluster"):
    cluster_labels = predict_rating_cluster(Number_of_Ratings, Weighted_Rating, Rating_category_encoder)
    if cluster_labels is not None:
        st.sidebar.success(f"Predicted Rating Cluster: {cluster_labels}")
    else:
        st.sidebar.error("Failed to make prediction")

# More Analysis (Add your custom analysis here)
st.header("Additional Analysis")
st.write("## K-Means Elbow ")
st.image('ELBOW Kmeans.png')

st.write("## DBSCAN Elbow ")
st.image('STAIRS DBSCAN.png')

st.write("## DBSCAN CLUSTER ")
st.image('CLUSTER DBSCAN.png')

# Footer
st.write("## ERROR")
st.image('error 1.png')
st.image('error 2.png')
st.image('error 3.png')