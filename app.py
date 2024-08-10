import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load the final data from the CSV file
file_path = 'final_data.csv'
final_df = pd.read_csv(file_path)

# Fill null values with 0
final_df = final_df.fillna(0)

# Streamlit App
st.title('Basic Reporting Environment')

# Filter by Description
description_options = final_df['Description'].unique()
selected_description = st.selectbox('Select Description', description_options)

# Filtered Data by Description
filtered_df = final_df[final_df['Description'] == selected_description]

# Filter by Location
location_options = filtered_df['Location'].unique()
selected_locations = st.multiselect('Select Location(s)', location_options, default=[location_options[0]])

# Filter by Metric
metric_options = filtered_df['Metric'].unique()
selected_metrics = st.multiselect('Select Metric(s)', metric_options, default=[metric_options[0]])

# Further filter the data based on selected locations and metrics
filtered_df = filtered_df[filtered_df['Location'].isin(selected_locations) & filtered_df['Metric'].isin(selected_metrics)]

# Set the index to Metric and Location for easier plotting
filtered_df = filtered_df.set_index(['Metric', 'Location'])

# Extract the time series data
time_series_data = filtered_df.loc[:, '01.01.2021':].T
time_series_data.index = pd.to_datetime(time_series_data.index, format='%d.%m.%Y')

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))

if len(selected_metrics) == 1:
    # When only one metric is selected, compare each year
    metric = selected_metrics[0]
    for location in selected_locations:
        for year in ['2021', '2022', '2023']:
            yearly_data = time_series_data[time_series_data.index.year == int(year)]
            ax.plot(yearly_data.index.month, yearly_data[(metric, location)], label=f'{metric} in {location} ({year})')
    ax.set_xticks(range(1, 13))
    ax.set
