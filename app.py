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
    ax.plot(future_index.strftime('%b'), y_pred, linestyle='--', label=f'2024 (Forecast) {label_suffix} ({location})')

def add_predictions_multiple(ax, series, location, label_suffix=""):
    model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=12)
    model_fit = model.fit()

    # Predict for 2024
    future_index = pd.date_range(start='2024-01-01', periods=12, freq='M').tz_localize(None)
    y_pred = model_fit.forecast(steps=12)

    # Plot the forecast data for 2024
    ax.plot(future_index, y_pred, linestyle='--', label=f'2024 (Forecast) {label_suffix} ({location})')

if len(selected_metrics) == 1:
    # If only one metric is selected, plot year-over-year comparison
    metric = selected_metrics[0]
    for location in selected_locations:
        data = time_series_data[(metric, location)]
        data_by_year = data.groupby(data.index.year)
        for year, year_data in data_by_year:
            ax.plot(year_data.index.strftime('%b'), year_data.values, label=f'{year} ({location})')

        if make_prediction:
            add_predictions(ax, data, location)
    
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
                add_predictions_multiple(ax, series, location, label_suffix=f'({metric})')

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

# Generate Insights for 2023 with Comparisons to Previous Years
st.subheader("Generated Insights for 2023")

def generate_insights_for_2023(data, metric, location):
    # Focus on 2023 data
    data_2023 = data[data.index.year == 2023]
    latest_value = data_2023[-1] if not data_2023.empty else None
    start_value = data_2023[0] if len(data_2023) > 1 else None
    mean_value = data_2023.mean()
    std_dev = data_2023.std()

    # Year-over-Year Comparisons
    previous_years = data[data.index.year < 2023]
    avg_previous_years = previous_years.groupby(previous_years.index.year).mean()
    avg_2023 = mean_value
    comparison_with_past = "higher" if avg_2023 > avg_previous_years.mean() else "lower"

    highest_prev_year = avg_previous_years.idxmax()[0]
    highest_prev_value = avg_previous_years.max()[0]

    # Monthly Trends in 2023
    highest_month = data_2023.idxmax().strftime('%B') if not data_2023.empty else "N/A"
    lowest_month = data_2023.idxmin().strftime('%B') if not data_2023.empty else "N/A"

    insights = [
        f"• In 2023, the {metric} for {location} started at {start_value:.2f} and ended at {latest_value:.2f}, showing a trend.",
        f"• The average value in 2023 was {avg_2023:.2f}, which is {comparison_with_past} than the average of previous years.",
        f"• The highest value observed in 2023 was in {highest_month}, and the lowest was in {lowest_month}.",
        f"• In previous years, the highest annual average was observed in {highest_prev_year} with a value of {highest_prev_value:.2f}."
    ]
    
    return insights

if len(selected_metrics) == 1:
    metric = selected_metrics[0]
    for location in selected_locations:
        data = time_series_data[(metric, location)]
        insights = generate_insights_for_2023(data, metric, location)
        for insight in insights:
            st.write(insight)
else:
    for metric in selected_metrics:
        for location in selected_locations:
            data = time_series_data[(metric, location)]
            insights = generate_insights_for_2023(data, metric, location)
            for insight in insights:
                st.write(insight)
