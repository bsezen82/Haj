import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import random

# Load the final data from the CSV file
file_path = 'final_data.csv'
final_df = pd.read_csv(file_path)

# Ensure column names are stripped of leading/trailing spaces
final_df.columns = final_df.columns.str.strip()

# Streamlit App
st.title('Advanced Reporting Environment')

# Filter by Report Metric
report_metric_options = final_df['Report Metric'].unique()
selected_report_metric = st.selectbox('Select Report Metric', report_metric_options)

# Filter by Analyses Metric
analyses_metric_options = {
    'General': None,
    'Destination Province': 'Destination Province',  # Replace 'B' with the actual column name for Destination Province
    'Origin': 'Origin',                 # Replace 'C' with the actual column name for Origin
    'Purpose of Visit': 'Purpose of Visit',       # Replace 'D' with the actual column name for Purpose of Visit
    'Mode of Arrival': 'Mode of Arrival',        # Replace 'E' with the actual column name for Mode of Arrival
    'Type of Accommodation': 'Type of Accommodation',  # Replace 'F' with the actual column name for Type of Accommodation
    'Spending Type': 'Spending Type'           # Replace 'G' with the actual column name for Spending Type
}
selected_analyses_metric = st.selectbox('Select Analyses Metric', analyses_metric_options.keys())

# Further filtering based on the Analyses Metric
column_name = analyses_metric_options[selected_analyses_metric]

# Apply filtering based on the selected report metric and analyses metric
if selected_analyses_metric == 'General':
    # If "General" is selected, filter all other columns as "All"
    for col_value in analyses_metric_options.values():
        if col_value:  # Only filter columns that are not None
            final_df = final_df[final_df[col_value] == 'All']
    filtered_df = final_df[final_df['Report Metric'] == selected_report_metric]
else:
    # If a specific analyses metric is selected
    filter_options = final_df[column_name].unique()
    selected_filters = st.multiselect(f'Select {selected_analyses_metric}', filter_options)
    
    # First, filter by selected report metric
    filtered_df = final_df[final_df['Report Metric'] == selected_report_metric]
    
    # Then filter other metrics as "All"
    for col_name, col_value in analyses_metric_options.items():
        if col_value and col_value != column_name:
            filtered_df = filtered_df[filtered_df[col_value] == 'All']
    
    # Finally, apply the selected filters for the chosen analyses metric
    filtered_df = filtered_df[filtered_df[column_name].isin(selected_filters)]


st.write(filtered_df)
