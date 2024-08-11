import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.signal import find_peaks
import numpy as np

# Load the final data from the CSV file
file_path = 'final_data.csv'
final_df = pd.read_csv(file_path)

# Fill null values with 0
final_df = final_df.fillna(0)

# Streamlit App
st.title('Advanced Reporting Environment with AI Insights')

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

# Option to make prediction
make_prediction = st.checkbox('Make predictions for 2024')

# Further filter the data based on selected locations and metrics
filtered_df = filtered_df[filtered_df['Location'].isin(selected_locations) & filtered_df['Metric'].isin(selected_metrics)]

# Set the index to Metric and Location for easier plotting
filtered_df = filtered_df.set_index(['Metric', 'Location'])

# Extract the time series data and ensure it's timezone-naive
time_series_data = filtered_df.loc[:, '01.01.2021':].T
time_series_data.index = pd.to_datetime(time_series_data.index, format='%d.%m.%Y', errors='coerce').tz_localize(None)

# Prepare the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Function to add 2024 predictions to the graph
def add_predictions(ax, series, location, label_suffix=""):
    model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=12)
    model_fit = model.fit()

    # Predict for 2024
    future_index = pd.date_range(start='2024-01-01', periods=12, freq='M').tz_localize(None)
    y_pred = model_fit.forecast(steps=12)

    # Plot the forecast data for 2024
    ax.plot(future_index, y_pred, linestyle='--', label=f'2024 (Forecast) {label_suffix} ({location})')
    return y_pred
    
def add_predictions_multiple(ax, series, location, label_suffix=""):
    model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=12)
    model_fit = model.fit()

    # Predict for 2024
    future_index = pd.date_range(start='2024-01-01', periods=12, freq='M').tz_localize(None)
    y_pred = model_fit.forecast(steps=12)

    # Plot the forecast data for 2024
    ax.plot(future_index, y_pred, linestyle='--', label=f'2024 (Forecast) {label_suffix} ({location})')

# Function to analyze trends, anomalies, and seasonality
def analyze_data(series, location):
    analysis = []
    trend = np.polyfit(range(len(series)), series, 1)
    trend_slope = trend[0]
    if trend_slope > 0:
        analysis.append(f"The data for {location} shows an upward trend.")
    else:
        analysis.append(f"The data for {location} shows a downward trend.")
    
    # Detect peaks (anomalies)
    peaks, _ = find_peaks(series, prominence=1)
    if len(peaks) > 0:
        analysis.append(f"Anomalies detected in {location}: Significant peaks at positions {peaks.tolist()}.")
    
    # Check for seasonality
    if series.index.freq or series.index.inferred_freq:
        analysis.append(f"Seasonality detected in the data for {location}.")
    
    return analysis

insights = []

if len(selected_metrics) == 1:
    # If only one metric is selected, plot year-over-year comparison
    metric = selected_metrics[0]
    for location in selected_locations:
        data = time_series_data[(metric, location)]
        data_by_year = data.groupby(data.index.year)
        for year, year_data in data_by_year:
            ax.plot(year_data.index.strftime('%b'), year_data.values, label=f'{year} ({location})')

            # Add AI-based insights
            avg_value = np.mean(year_data.values)
            insights.append(f"In {year}, the average value for {metric} in {location} was {avg_value:.2f}.")
            insights.extend(analyze_data(year_data.values, location))
            
        if make_prediction:
            y_pred = add_predictions(ax, data, location)
            insights.append(f"Predictions for 2024 suggest that the trend will continue with a similar pattern, reaching an estimated value of {y_pred[-1]:.2f} by December 2024.")
    
    ax.set_xlabel('Month')
    ax.set_xticks(range(12))
    ax.set_xticklabels([pd.to_datetime(f'{i+1}', format='%m').strftime('%b') for i in range(12)])
else:
    # If more than one metric is selected, plot each metric over time
    for metric in selected_metrics:
        for location in selected_locations:
            series = time_series_data[(metric, location)]
            ax.plot(series.index, series, label=f'{metric} ({location})')

            if make_prediction:
                y_pred = add_predictions_multiple(ax, series, location, label_suffix=f'({metric})')
                insights.append(f"The forecast for {metric} in {location} shows an expected increase to {y_pred[-1]:.2f} by December 2024.")
            
            # Analyze trends, anomalies, and seasonality
            insights.extend(analyze_data(series.values, location))

    # Extend the x-axis to cover the prediction period if predictions are made
    if make_prediction:
        ax.set_xlim([time_series_data.index.min(), pd.to_datetime('2024-12-31')])
    ax.set_xlabel('Date')

# Set axis labels and title
ax.set_ylabel('Value')
ax.set_title('Time Series Trend with 2024 Predictions')
ax.legend(title='Metric and Location')

# Display the plot
st.pyplot(fig)

# Display AI-based insights
st.header('AI-Generated Insights')
for insight in insights:
    st.write(insight)
