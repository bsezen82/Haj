import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import numpy as np

# Ensure the cmdstanpy backend is used for Prophet
import cmdstanpy
cmdstanpy.install_cmdstan()


# Load the final data from the CSV file
file_path = 'final_data.csv'
final_df = pd.read_csv(file_path)

# Fill null values with 0
final_df = final_df.fillna(0)

# Streamlit App
st.title('Haj and Umrah Statistics')

# Filter by Description
description_options = final_df['Description'].unique()
selected_description = st.selectbox('Select Report', description_options)

# Filtered Data by Description
filtered_df = final_df[final_df['Description'] == selected_description]

# Filter by Location
location_options = filtered_df['Location'].unique()
selected_locations = st.multiselect('Select Location(s)', location_options, default=[location_options[0]])

# Filter by Metric
metric_options = filtered_df['Metric'].unique()
selected_metrics = st.multiselect('Select Metric(s)', metric_options, default=[metric_options[0]])

# Option to make prediction
make_prediction = st.checkbox('Make prediction for next year')

# Further filter the data based on selected locations and metrics
filtered_df = filtered_df[filtered_df['Location'].isin(selected_locations) & filtered_df['Metric'].isin(selected_metrics)]

# Set the index to Metric and Location for easier plotting
filtered_df = filtered_df.set_index(['Metric', 'Location'])

# Extract the time series data
time_series_data = filtered_df.loc[:, '01.01.2021':].T
time_series_data.index = pd.to_datetime(time_series_data.index, format='%d.%m.%Y')

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))
for column in time_series_data.columns:
    ax.plot(time_series_data.index, time_series_data[column], label=column)

if make_prediction:
    st.header('Forecasting for 2024')
    for metric in selected_metrics:
        for location in selected_locations:
            series = time_series_data[(metric, location)]
            df = pd.DataFrame({'ds': series.index, 'y': series.values})
            model = Prophet()
            model.fit(df)
            future = model.make_future_dataframe(periods=12, freq='M')
            forecast = model.predict(future)
            forecast.set_index('ds', inplace=True)

            # Plot the real data
            ax.plot(series.index, series, label=f'{metric} in {location} (Actual)')
            
            # Plot the forecast data only for 2024
            forecast_2024 = forecast.loc[forecast.index >= '2024-01-01']
            ax.plot(forecast_2024.index, forecast_2024['yhat'], linestyle='--', label=f'{metric} in {location} (Forecast)')
            ax.fill_between(forecast_2024.index, forecast_2024['yhat_lower'], forecast_2024['yhat_upper'], alpha=0.3)

ax.set_title('Time Series Trend')
ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.legend(title='Metric and Location')
st.pyplot(fig)
