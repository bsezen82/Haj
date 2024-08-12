import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import random

# Load the final data from the CSV file
file_path = '/mnt/data/final_data.csv'
final_df = pd.read_csv(file_path)

# Fill null values with 0
final_df = final_df.fillna(0)

# Streamlit App
st.title('Enhanced Reporting Environment')

# Filter by Report Metric
report_metric_options = final_df['report metric'].unique()
selected_report_metric = st.selectbox('Select Report Metric', report_metric_options)

# Filter by Analyses Metric
analysis_metric_options = ['general', 'destination province', 'origin', 'purpose of visit', 'mode of arrival', 'type of accommodation', 'spending type']
selected_analysis_metric = st.selectbox('Select Analyses Metric', analysis_metric_options)

# Further filtering based on the selected Analyses Metric
if selected_analysis_metric == 'destination province':
    destination_province_options = final_df['B'].unique()
    selected_destination_provinces = st.multiselect('Select Destination Province(s)', destination_province_options)
    filtered_df = final_df[(final_df['report metric'] == selected_report_metric) & (final_df['B'].isin(selected_destination_provinces))]

elif selected_analysis_metric == 'origin':
    origin_options = final_df['C'].unique()
    selected_origins = st.multiselect('Select Origin(s)', origin_options)
    filtered_df = final_df[(final_df['report metric'] == selected_report_metric) & (final_df['C'].isin(selected_origins))]

elif selected_analysis_metric == 'purpose of visit':
    purpose_options = final_df['D'].unique()
    selected_purposes = st.multiselect('Select Purpose(s) of Visit', purpose_options)
    filtered_df = final_df[(final_df['report metric'] == selected_report_metric) & (final_df['D'].isin(selected_purposes))]

elif selected_analysis_metric == 'mode of arrival':
    arrival_options = final_df['E'].unique()
    selected_arrivals = st.multiselect('Select Mode(s) of Arrival', arrival_options)
    filtered_df = final_df[(final_df['report metric'] == selected_report_metric) & (final_df['E'].isin(selected_arrivals))]

elif selected_analysis_metric == 'type of accommodation':
    accommodation_options = final_df['F'].unique()
    selected_accommodations = st.multiselect('Select Type(s) of Accommodation', accommodation_options)
    filtered_df = final_df[(final_df['report metric'] == selected_report_metric) & (final_df['F'].isin(selected_accommodations))]

elif selected_analysis_metric == 'spending type':
    spending_options = final_df['G'].unique()
    selected_spending = st.multiselect('Select Spending Type(s)', spending_options)
    filtered_df = final_df[(final_df['report metric'] == selected_report_metric) & (final_df['G'].isin(selected_spending))]

else:
    # No further filtering for 'general'
    filtered_df = final_df[final_df['report metric'] == selected_report_metric]

# Set the index to the appropriate columns for easier plotting
filtered_df = filtered_df.set_index(['report metric', 'B', 'C', 'D', 'E', 'F', 'G'])

# Extract the time series data and ensure it's timezone-naive
time_series_data = filtered_df.loc[:, '01.01.2021':].T
time_series_data.index = pd.to_datetime(time_series_data.index, format='%d.%m.%Y', errors='coerce').tz_localize(None)

# Prepare the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Option to make prediction
make_prediction = st.checkbox('Make predictions for 2024')

# Function to add 2024 predictions to the graph
def add_predictions(ax, series, label_suffix=""):
    model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=12)
    model_fit = model.fit()

    # Predict for 2024
    future_index = pd.date_range(start='2024-01-01', periods=12, freq='M').tz_localize(None)
    y_pred = model_fit.forecast(steps=12)

    # Plot the forecast data for 2024
    ax.plot(future_index.strftime('%b'), y_pred, linestyle='--', label=f'2024 (Forecast) {label_suffix}')

# Plot the selected data
for index in filtered_df.columns:
    series = filtered_df[index]
    ax.plot(series.index, series, label=f'{index}')

    if make_prediction:
        add_predictions(ax, series, label_suffix=f'({index})')

ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.set_title(f'Time Series Trend for {selected_report_metric}')
ax.legend(title='Metrics and Selections')

# Display the plot
st.pyplot(fig)

# Generate Insights for 2023 and Multiple Metric Comparisons
st.subheader("Generated Insights for 2023")

def generate_insights_for_2023(data, metric_name):
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

    highest_prev_year = avg_previous_years.idxmax()
    highest_prev_value = avg_previous_years.max()

    # Monthly Trends in 2023
    highest_month = data_2023.idxmax().strftime('%B') if not data_2023.empty else "N/A"
    lowest_month = data_2023.idxmin().strftime('%B') if not data_2023.empty else "N/A"

    insights = [
        f"• For {metric_name} in 2023, the year started at {start_value:.2f} and ended at {latest_value:.2f}, showing a trend.",
        f"• The average value of {metric_name} in 2023 was {avg_2023:.2f}, which is {comparison_with_past} than the average of previous years.",
        f"• The highest value for {metric_name} in 2023 was observed in {highest_month}, and the lowest was in {lowest_month}.",
        f"• In previous years, the highest annual average for {metric_name} was observed in {highest_prev_year} with a value of {highest_prev_value:.2f}."
    ]
    
    return insights

def generate_comparative_insights(data_dict):
    insights = []

    # Compare averages
    avg_values = {metric: data.mean() for metric, data in data_dict.items()}
    max_avg_metric = max(avg_values, key=avg_values.get)
    min_avg_metric = min(avg_values, key=avg_values.get)
    insights.append(f"• The metric with the highest average value is {max_avg_metric} with {avg_values[max_avg_metric]:.2f}.")
    insights.append(f"• The metric with the lowest average value is {min_avg_metric} with {avg_values[min_avg_metric]:.2f}.")

    # Compare trends
    trend_changes = {metric: data[-1] - data[0] for metric, data in data_dict.items()}
    increasing_metric = max(trend_changes, key=trend_changes.get)
    decreasing_metric = min(trend_changes, key=trend_changes.get)
    insights.append(f"• The metric that increased the most over the period is {increasing_metric}.")
    insights.append(f"• The metric that decreased the most over the period is {decreasing_metric}.")

    # Compare volatility
    volatility = {metric: data.std() for metric, data in data_dict.items()}
    most_volatile_metric = max(volatility, key=volatility.get)
    least_volatile_metric = min(volatility, key=volatility.get)
    insights.append(f"• The most volatile metric is {most_volatile_metric}.")
    insights.append(f"• The least volatile metric is {least_volatile_metric}.")

    return insights

# Collect all insights
all_insights = []

if len(selected_metrics) == 1:
    metric_name = selected_report_metric
    for index in filtered_df.columns:
        data = filtered_df[index]
        insights = generate_insights_for_2023(data, metric_name)
        all_insights.extend(insights)
else:
    data_dict = {}
    metric_name = selected_report_metric
    for index in filtered_df.columns:
        data_dict[index] = filtered_df[index]
        insights = generate_insights_for_2023(filtered_df[index], metric_name)
        all_insights.extend(insights)

    # Generate comparative insights for multiple metrics
    comparative_insights = generate_comparative_insights(data_dict)
    all_insights.extend(comparative_insights)

# Limit the total number of insights to 5
max_insights = 5
if len(all_insights) > max_insights:
    all_insights = random.sample(all_insights, max_insights)
