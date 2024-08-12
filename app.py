import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import random

# Load the final data from the CSV file
file_path = 'final_data.csv'
final_df = pd.read_csv(file_path)

# Fill null values with 0
final_df = final_df.fillna(0)

# Streamlit App
st.title('Haj and Umrah Analytics')

# Filter by Report Metric
report_metric_options = final_df['Report Metric'].unique()
selected_report_metric = st.selectbox('Select Report Metric', report_metric_options)

# Filter by Analyses Metric
analyses_metric_options = ['General', 'Destination Province', 'Origin', 'Purpose of Visit', 'Mode of Arrival', 'Type of Accommodation', 'Spending Type']
selected_analyses_metric = st.selectbox('Select Analyses Metric', analyses_metric_options)

# Further filtering based on the Analyses Metric
if selected_analyses_metric == 'General':
    filtered_df = final_df[final_df['Report Metric'] == selected_report_metric]
elif selected_analyses_metric == 'Destination Province':
    destination_options = final_df['B'].unique()
    selected_destinations = st.multiselect('Select Destination Province(s)', destination_options)
    filtered_df = final_df[(final_df['Report Metric'] == selected_report_metric) & (final_df['B'].isin(selected_destinations))]
elif selected_analyses_metric == 'Origin':
    origin_options = final_df['C'].unique()
    selected_origins = st.multiselect('Select Origin(s)', origin_options)
    filtered_df = final_df[(final_df['Report Metric'] == selected_report_metric) & (final_df['C'].isin(selected_origins))]
elif selected_analyses_metric == 'Purpose of Visit':
    purpose_options = final_df['D'].unique()
    selected_purposes = st.multiselect('Select Purpose(s) of Visit', purpose_options)
    filtered_df = final_df[(final_df['Report Metric'] == selected_report_metric) & (final_df['D'].isin(selected_purposes))]
elif selected_analyses_metric == 'Mode of Arrival':
    mode_options = final_df['E'].unique()
    selected_modes = st.multiselect('Select Mode(s) of Arrival', mode_options)
    filtered_df = final_df[(final_df['Report Metric'] == selected_report_metric) & (final_df['E'].isin(selected_modes))]
elif selected_analyses_metric == 'Type of Accommodation':
    accommodation_options = final_df['F'].unique()
    selected_accommodations = st.multiselect('Select Type(s) of Accommodation', accommodation_options)
    filtered_df = final_df[(final_df['Report Metric'] == selected_report_metric) & (final_df['F'].isin(selected_accommodations))]
elif selected_analyses_metric == 'Spending Type':
    spending_options = final_df['G'].unique()
    selected_spending = st.multiselect('Select Spending Type(s)', spending_options)
    filtered_df = final_df[(final_df['Report Metric'] == selected_report_metric) & (final_df['G'].isin(selected_spending))]

# Option to make prediction
make_prediction = st.checkbox('Make predictions for 2024')

# Set the index to ensure correct filtering and plotting
filtered_df = filtered_df.set_index(['report metric'] + [selected_analyses_metric])

# Extract the time series data and ensure it's timezone-naive
time_series_data = filtered_df.loc[:, '01.01.2021':].T
time_series_data.index = pd.to_datetime(time_series_data.index, format='%d.%m.%Y', errors='coerce').tz_localize(None)

# Prepare the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Function to add 2024 predictions to the graph
def add_predictions(ax, series, label_suffix=""):
    model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=12)
    model_fit = model.fit()

    # Predict for 2024
    future_index = pd.date_range(start='2024-01-01', periods=12, freq='M').tz_localize(None)
    y_pred = model_fit.forecast(steps=12)

    # Plot the forecast data for 2024
    ax.plot(future_index.strftime('%b'), y_pred, linestyle='--', label=f'2024 (Forecast) {label_suffix}')

if len(filtered_df.index.get_level_values(selected_analyses_metric).unique()) == 1:
    # Single value selected in the third filtering
    selected_value = filtered_df.index.get_level_values(selected_analyses_metric)[0]
    data = time_series_data[selected_value]
    data_by_year = data.groupby(data.index.year)
    for year, year_data in data_by_year:
        ax.plot(year_data.index.strftime('%b'), year_data.values, label=f'{year} ({selected_value})')

    if make_prediction:
        add_predictions(ax, data, label_suffix=selected_value)
    
    ax.set_xlabel('Month')
    ax.set_xticks(range(12))
    ax.set_xticklabels([pd.to_datetime(f'{i+1}', format='%m').strftime('%b') for i in range(12)])
else:
    # Multiple values selected in the third filtering
    for value in filtered_df.index.get_level_values(selected_analyses_metric).unique():
        series = time_series_data[value]
        ax.plot(series.index, series, label=f'{value}')

        if make_prediction:
            add_predictions(ax, series, label_suffix=value)

    # Extend the x-axis to cover the prediction period if predictions are made
    if make_prediction:
        ax.set_xlim([time_series_data.index.min(), pd.to_datetime('2024-12-31')])
    ax.set_xlabel('Date')

# Set axis labels and title
ax.set_ylabel('Value')
ax.set_title(f'Trend Analysis: {selected_report_metric} with {selected_analyses_metric}')
ax.legend(title=f'{selected_analyses_metric}')

# Display the plot
st.pyplot(fig)

# Generate Insights for 2023
st.subheader("Generated Insights for 2023")

def generate_insights_for_2023(data, metric, analysis_value):
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
        f"• For {metric} in {analysis_value}, 2023 started at {start_value:.2f} and ended at {latest_value:.2f}, showing a trend.",
        f"• The average value of {metric} in 2023 was {avg_2023:.2f}, which is {comparison_with_past} than the average of previous years.",
        f"• The highest value for {metric} in 2023 was observed in {highest_month}, and the lowest was in {lowest_month}.",
        f"• In previous years, the highest annual average for {metric} was observed in {highest_prev_year} with a value of {highest_prev_value:.2f}."
    ]
    
    return insights

# Generate insights based on the third filtering step
if len(filtered_df.index.get_level_values(selected_analyses_metric).unique()) == 1:
    # Single value selected in the third filtering
    selected_value = filtered_df.index.get_level_values(selected_analyses_metric)[0]
    data = time_series_data[selected_value]
    insights = generate_insights_for_2023(data, selected_report_metric, selected_value)
else:
    # Multiple values selected in the third filtering
    insights = []
    for value in filtered_df.index.get_level_values(selected_analyses_metric).unique():
        data = time_series_data[value]
        insights.extend(generate_insights_for_2023(data, selected_report_metric, value))

# Limit the total number of insights to 5
max_insights = 5
if len(insights) > max_insights:
    insights = random.sample(insights, max_insights)

# Display the limited insights
for insight in insights:
    st.write(insight)
