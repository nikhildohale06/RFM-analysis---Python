import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import pickle

with open('df.pickle','rb') as pkl:
    data = pickle.load(pkl)
    
  
df = pd.read_csv("rfm_data.csv")
df.head()

from datetime import datetime

# Assuming you have a DataFrame named df with your dataset

# Convert 'PurchaseDate' column to datetime type
df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])

# Get the current date
current_date = datetime.now()

# Calculate recency for each customer
df['Recency'] = (current_date - df['PurchaseDate']).dt.days
# Assuming you have a DataFrame named df with your dataset

# Frequency: Count the number of transactions per CustomerID
frequency_df = df.groupby('CustomerID').size().reset_index(name='Frequency')

# Monetary: Sum TransactionAmount per CustomerID
monetary_df = df.groupby('CustomerID')['TransactionAmount'].sum().reset_index(name='Monetary')

# Merge frequency and monetary DataFrames with the original DataFrame
df = pd.merge(df, frequency_df, on='CustomerID', how='left')
df = pd.merge(df, monetary_df, on='CustomerID', how='left')
df.head()

# Define scoring criteria for each RFM value
recency_scores = [5, 4, 3, 2, 1]  # Higher score for lower recency (more recent)
frequency_scores = [1, 2, 3, 4, 5]  # Higher score for higher frequency
monetary_scores = [1, 2, 3, 4, 5]  # Higher score for higher monetary value

# Calculate RFM scores
df['RecencyScore'] = pd.cut(df['Recency'], bins=5, labels=recency_scores)
df['FrequencyScore'] = pd.cut(df['Frequency'], bins=5, labels=frequency_scores)
df['MonetaryScore'] = pd.cut(df['Monetary'], bins=5, labels=monetary_scores)

df['RecencyScore'] = df['RecencyScore'].astype(int)
df['FrequencyScore'] = df['FrequencyScore'].astype(int)
df['MonetaryScore'] = df['MonetaryScore'].astype(int)

# Calculate RFM score by combining the individual scores
df['RFM_Score'] = df['RecencyScore'] + df['FrequencyScore'] + df['MonetaryScore']

# Create RFM segments based on the RFM score
segment_labels = ['Low-Value', 'Mid-Value', 'High-Value']
df['Value Segment'] = pd.qcut(df['RFM_Score'], q=3, labels=segment_labels)

# Create a new column for RFM Customer Segments
df['RFM Customer Segments'] = ''

# Assign RFM segments based on the RFM score
df.loc[df['RFM_Score'] >= 9, 'RFM Customer Segments'] = 'Champions'
df.loc[(df['RFM_Score'] >= 6) & (df['RFM_Score'] < 9), 'RFM Customer Segments'] = 'Potential Loyalists'
df.loc[(df['RFM_Score'] >= 5) & (df['RFM_Score'] < 6), 'RFM Customer Segments'] = 'At Risk Customers'
df.loc[(df['RFM_Score'] >= 4) & (df['RFM_Score'] < 5), 'RFM Customer Segments'] = "Can't Lose"
df.loc[(df['RFM_Score'] >= 3) & (df['RFM_Score'] < 4), 'RFM Customer Segments'] = "Lost"

import streamlit as st
def main():
    st.title("RFM Analysis Dashboard")
    st.write("Analyze customer segments based on RFM scores.")
    
    # Dropdown for selecting the chart
    chart_type = st.selectbox("Select Chart Type", [
        'RFM Value Segment Distribution',
        'Distribution of RFM Values within Customer Segment',
        'Correlation Matrix of RFM Values within Champions Segment',
        'Comparison of RFM Segments',
        'Comparison of RFM Segments based on Scores',
        'Distribution of RFM values within Champions Segment'
    ])
    
     # Display selected chart
    if chart_type == 'RFM Value Segment Distribution':
        st.pyplot(display_segment_dist())
    elif chart_type == 'Distribution of RFM Values within Customer Segment':
        st.pyplot(display_treemap_segment_product())
    elif chart_type == 'Correlation Matrix of RFM Values within Champions Segment':
        st.pyplot(display_corr_heatmap())
    elif chart_type == 'Comparison of RFM Segments':
        st.pyplot(display_segment_comparison())
    elif chart_type == 'Comparison of RFM Segments based on Scores':
        st.pyplot(display_segment_scores())
    elif chart_type == 'Distribution of RFM values within Champions Segment':
        st.pyplot(rfm_values_dist_champions_data())

def display_segment_dist():
    df['Value Segment'] = pd.qcut(df['RFM_Score'], q=3, labels=segment_labels)
    segment_counts = df['Value Segment'].value_counts().reset_index()
    segment_counts.columns = ['Value Segment', 'Count']
    pastel_colors = ['#FFB6C1', '#FFD700', '#98FB98', '#ADD8E6', '#FFA07A']
    plt.figure(figsize=(10, 6))
    plt.bar(segment_counts['Value Segment'], segment_counts['Count'], color=pastel_colors)
    plt.title('RFM Value Segment Distribution')
    plt.xlabel('RFM Value Segment')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.show()
    

# Function to display Distribution of RFM Values within Customer Segment using Matplotlib
def display_treemap_segment_product():
    segment_product_counts = df.groupby(['Value Segment', 'RFM Customer Segments']).size().reset_index(name='Count')

    segment_product_counts = segment_product_counts.sort_values('Count', ascending=False)

    fig_treemap_segment_product = px.treemap(segment_product_counts, 
                                         path=['Value Segment', 'RFM Customer Segments'], 
                                         values='Count',
                                         color='Value Segment', color_discrete_sequence=px.colors.qualitative.Pastel,
                                         title='RFM Customer Segments by Value')
    fig_treemap_segment_product.show()

    

# Function to display Correlation Matrix of RFM Values within Champions Segment using Seaborn
def display_corr_heatmap():
    champions_segment = df[df['RFM Customer Segments'] == 'Champions']
    correlation_matrix = champions_segment[['RecencyScore', 'FrequencyScore', 'MonetaryScore']].corr()

# Create heatmap using Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='RdBu', vmin=-1, vmax=1)
    plt.title('Correlation Matrix of RFM Values within Champions Segment')
    plt.xlabel('RFM Value')
    plt.ylabel('RFM Value')
    plt.show()

   

# Function to display Comparison of RFM Segments using Matplotlib
def display_segment_comparison():
    
    # Calculate segment counts
    segment_counts = df['RFM Customer Segments'].value_counts()

# Define colors
    pastel_colors = [(0.984, 0.705, 0.682), (0.702, 0.871, 0.882), (0.769, 0.855, 0.678), (0.988, 0.816, 0.705), (0.965, 0.725, 0.792)]
    champions_color = (0.620, 0.789, 0.882)

# Create the bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(segment_counts.index, segment_counts.values, color=[champions_color if segment == 'Champions' else pastel_colors[i] for i, segment in enumerate(segment_counts.index)])

# Update the layout
    plt.title('Comparison of RFM Segments')
    plt.xlabel('RFM Segments')
    plt.ylabel('Number of Customers')

# Show the bar chart
    plt.xticks(rotation=45, ha='right')
    plt.show()

    

# Function to display Comparison of RFM Segments based on Scores using Matplotlib
def display_segment_scores():
    # Write your Matplotlib code here to create Comparison of RFM Segments based on Scores
    # Calculate the average Recency, Frequency, and Monetary scores for each segment
    segment_scores = df.groupby('RFM Customer Segments')[['RecencyScore', 'FrequencyScore', 'MonetaryScore']].mean().reset_index()

# Define colors
    colors = [(0.246,0.468,0.680),(0.135,0.357,0.579),(0.467,0.857,0.698)]

# Melt the dataframe for Seaborn
    melted_segment_scores = segment_scores.melt(id_vars='RFM Customer Segments', var_name='Score Type', value_name='Score')

# Create the grouped bar chart using Seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(x='RFM Customer Segments', y='Score', hue='Score Type', data=melted_segment_scores, palette=colors)
    plt.title('Comparison of RFM Segments based on Recency, Frequency, and Monetary Scores')
    plt.xlabel('RFM Segments')
    plt.ylabel('Score')
    plt.legend(title='Score Type')
    plt.xticks(rotation=45)
    plt.show()


def rfm_values_dist_champions_data():
# Filter the data to include only the customers in the Champions segment
    champions_segment = df[df['RFM Customer Segments'] == 'Champions']

# Create box plots using Matplotlib
    plt.figure(figsize=(10, 6))
    plt.boxplot([champions_segment['RecencyScore'], champions_segment['FrequencyScore'], champions_segment['MonetaryScore']],
            labels=['Recency', 'Frequency', 'Monetary'])
    plt.title('Distribution of RFM Values within Champions Segment')
    plt.xlabel('RFM Value')
    plt.ylabel('Score')
    plt.grid(True)
    plt.show()

    

if __name__ == '__main__':
    main()       