import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

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

# Extract the time series data and ensure it's timezone-naive
time_series_data = filtered_df.loc[:, '01.01.2021':].T
time_series_data.index = pd.to_datetime(time_series_data.index, format='%d.%m.%Y', errors='coerce').tz_localize(None)

# Prepare the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Loop through each selected metric and location to plot actual data and predictions
for metric in selected_metrics:
    for location in selected_locations:
        # Get the series for the current metric and location
        series = time_series_data[(metric, location)]

        # Apply Exponential Smoothing to predict for 2024
        model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=12)
        model_fit = model.fit()

        # Predict for 2024
        future_index = pd.date_range(start='2024-01-01', periods=12, freq='M').tz_localize(None)
        y_pred = model_fit.forecast(steps=12)

        # Plot the actual data for 2021-2023
        ax.plot(series.index, series, label=f'{metric} in {location} (Actual)')

        # Plot the forecast data for 2024
        ax.plot(future_index, y_pred, linestyle='--', label=f'{metric} in {location} (Forecast)')

# Set axis labels and title
ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.set_title('Time Series Trend with 2024 Predictions')
ax.legend(title='Metric and Location')

# Display the plot
st.pyplot(fig)
