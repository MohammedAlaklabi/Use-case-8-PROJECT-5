import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Title and introduction
st.title("Restaurant Ratings Analysis")
st.write("""
## Introduction
Restaurant ratings serve as a valuable reference for both consumers and restaurants. Restaurant ratings influence how much money a restaurant makes and help customers choose where to eat.
""")

# Load the data
@st.cache
def load_data():
    data = pd.read_csv('FINALDATA.csv')  # Update this with the path to your data file
    data = data.drop('Unnamed: 0', axis=1)
    return data

data = load_data()

# Display the data
st.write("## Data Overview")
st.dataframe(data.head())


# Visualization: Distribution of Ratings
st.write("## Distribution of Ratings")
rating_counts = data['Rating'].value_counts()
fig = px.bar(rating_counts, x=rating_counts.index, y=rating_counts.values, labels={'x': 'Rating', 'y': 'Count'}, title='Distribution of Ratings')
st.plotly_chart(fig)


# Visualization: Scatter Plot of Ratings vs. Number of Ratings
st.write("## Ratings vs. Number of Ratings (Interactive)")
fig = px.scatter(data, x='Number of Ratings', y='Rating', title='Ratings vs. Number of Ratings', labels={'Number of Ratings': 'Number of Ratings', 'Rating': 'Rating'})
st.plotly_chart(fig)

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


