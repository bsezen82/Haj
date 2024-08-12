import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import random

# Load the updated final data from the CSV file
file_path = '/mnt/data/final_data.csv'
final_df = pd.read_csv(file_path)

# Fill null values with 0
final_df = final_df.fillna(0)

# Streamlit App
st.title('Enhanced Reporting Environment')

# Filter by Report Metric
metric_options = final_df['report metric'].unique()
selected_metric = st.selectbox('Select Report Metric', metric_options)

# Filtered Data by Report Metric
filtered_df = final_df[final_df['report metric'] == selected_metric]

# Filter by Destination Province
province_options = filtered_df['Destination Province'].unique()
selected_provinces = st.multiselect('Select Destination Province(s)', province_options, default=province_options)

# Filter by Origin
origin_options = filtered_df['Origin'].unique()
selected_origins = st.multiselect('Select Origin(s)', origin_options, default=origin_options)

# Filter by Purpose of Visit
purpose_options = filtered_df['Purpose of Visit'].unique()
selected_purposes = st.multiselect('Select Purpose of Visit(s)', purpose_options, default=purpose_options)

# Filter by Mode of Arrival
arrival_options = filtered_df['Mode of Arrival'].unique()
selected_arrivals = st.multiselect('Select Mode of Arrival(s)', arrival_options, default=arrival_options)

# Filter by Type of Accommodation
accommodation_options = filtered_df['Type of Accommodation'].unique()
selected_accommodations = st.multiselect('Select Type of Accommodation(s)', accommodation_options, default=accommodation_options)

# Filter by Spending Types
spending_options = filtered_df['Spending Types'].unique()
selected_spending = st.multiselect('Select Spending Type(s)', spending_options, default=spending_options)

# Apply all filters to the DataFrame
filtered_df = filtered_df[
    (filtered_df['Destination Province'].isin(selected_provinces)) &
    (filtered_df['Origin'].isin(selected_origins)) &
    (filtered_df['Purpose of Visit'].isin(selected_purposes)) &
    (filtered_df['Mode of Arrival'].isin(selected_arrivals)) &
    (filtered_df['Type of Accommodation'].isin(selected_accommodations)) &
    (filtered_df['Spending Types'].isin(selected_spending))
]

# Set the index to Metric and Location for easier plotting
filtered_df = filtered_df.set_index(['report metric', 'Destination Province', 'Origin', 'Purpose of Visit', 'Mode of Arrival', 'Type of Accommodation', 'Spending Types'])

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

if len(selected_metric) == 1:
    for index, row in filtered_df.iterrows():
        series = row['01.01.2021':]
        ax.plot(time_series_data.index, series, label=f'{index}')

        # If the user opts to make predictions for 2024
        if st.checkbox('Make predictions for 2024'):
            add_predictions(ax, series, label_suffix=f'{index}')

# Set axis labels and title
ax.set_xlabel('Month')
ax.set_ylabel('Value')
ax.set_title(f'Time Series Trend for {selected_metric}')
ax.legend(title='Metric Details')

# Display the plot
st.pyplot(fig)

# Generate Insights for 2023
st.subheader("Generated Insights for 2023")

def generate_insights_for_2023(data, index_label):
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
        f"• For {index_label}, 2023 started at {start_value:.2f} and ended at {latest_value:.2f}, showing a trend.",
        f"• The average value in 2023 was {avg_2023:.2f}, which is {comparison_with_past} than the average of previous years.",
        f"• The highest value in 2023 was observed in {highest_month}, and the lowest was in {lowest_month}.",
        f"• In previous years, the highest annual average was observed in {highest_prev_year} with a value of {highest_prev_value:.2f}."
    ]
    
    return insights

all_insights = []

# Generate insights for the selected metric
for index, row in filtered_df.iterrows():
    series = row['01.01.2021':]
    insights = generate_insights_for_2023(series, index)
    all_insights.extend(insights)

# Limit the total number of insights to 5
max_insights = 5
if len(all_insights) > max_insights:
    all_insights = random.sample(all_insights, max_insights)

# Display the limited insights
for insight in all_insights:
    st.write(insight)
