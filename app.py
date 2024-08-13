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
selected_report_metric = st.selectbox('Select Report Metric', report_metric_options, index=list(report_metric_options).index("Number of Tourists"))

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

# Filter other metrics as "All"
if column_name is not None:
    for col_name, col_value in analyses_metric_options.items():
        if col_value and col_value != column_name:
            final_df = final_df[final_df[col_value] == 'All']
else:
    # If "General" is selected, filter all columns as "All"
    for col_value in analyses_metric_options.values():
        if col_value:
            final_df = final_df[final_df[col_value] == 'All']

# Apply filtering based on the selected report metric and analyses metric
if column_name is None:
    filtered_df = final_df[final_df['Report Metric'] == selected_report_metric]
else:
    filter_options = final_df[column_name].unique()
    selected_filters = st.multiselect(f'Select {selected_analyses_metric}', filter_options, default="All")
    filtered_df = final_df[(final_df['Report Metric'] == selected_report_metric) & (final_df[column_name].isin(selected_filters))]

# Option to make prediction
make_prediction = st.checkbox('Make predictions for 2024')

# Check if filtered_df is empty or missing expected indices
if filtered_df.empty:
    st.warning(f"No data available for the selected filters.")
else:
    # Set the index for correct filtering and plotting
    if column_name is None:
        filtered_df = filtered_df.set_index('Report Metric')
    else:
        filtered_df = filtered_df.set_index(['Report Metric', column_name])

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

    # Convert future_index to the correct format for plotting
    future_index = pd.to_datetime(future_index).strftime('%Y-%m-%d')

    # Plot the forecast data for 2024
    ax.plot(pd.to_datetime(future_index), y_pred, linestyle='--', label=f'2024 (Forecast) {label_suffix}')
        

    # Plotting logic based on how many values are selected in the third filtering
    if selected_analyses_metric == 'General':
        # Handle the case where "General" is selected
        if selected_report_metric in time_series_data.columns:
            data = time_series_data[selected_report_metric]
            data_by_year = data.groupby(data.index.year)
            for year, year_data in data_by_year:
                ax.plot(year_data.index.strftime('%b'), year_data.values, label=f'{year} ({selected_report_metric})')

            if make_prediction:
                add_predictions(ax, data, label_suffix=selected_report_metric)
            
            ax.set_xlabel('Month')
            ax.set_xticks(range(12))
            ax.set_xticklabels([pd.to_datetime(f'{i+1}', format='%m').strftime('%b') for i in range(12)])
        else:
            st.warning(f"Selected value '{selected_report_metric}' not found in the data.")
    elif len(filtered_df.index.get_level_values(column_name).unique()) == 1:
        # Single value selected in the third filtering
        selected_value = filtered_df.index.get_level_values(column_name)[0] if column_name else selected_report_metric
        
        # Check if the combination exists in the time_series_data columns
        if (selected_report_metric, selected_value) in time_series_data.columns:
            data = time_series_data[(selected_report_metric, selected_value)]
            data_by_year = data.groupby(data.index.year)
            for year, year_data in data_by_year:
                ax.plot(year_data.index.strftime('%b'), year_data.values, label=f'{year} ({selected_value})')

            if make_prediction:
                add_predictions(ax, data, label_suffix=selected_value)
            
            ax.set_xlabel('Month')
            ax.set_xticks(range(12))
            ax.set_xticklabels([pd.to_datetime(f'{i+1}', format='%m').strftime('%b') for i in range(12)])
        else:
            st.warning(f"Selected combination '{selected_report_metric}, {selected_value}' not found in the data.")
    else:
        # Multiple values selected in the third filtering
        for value in filtered_df.index.get_level_values(column_name).unique():
            # Check if the combination exists in the time_series_data columns
            if (selected_report_metric, value) in time_series_data.columns:
                series = time_series_data[(selected_report_metric, value)]
                ax.plot(series.index, series, label=f'{value}')

                if make_prediction:
                    add_predictions(ax, series, label_suffix=value)
            else:
                st.warning(f"Selected combination '{selected_report_metric}, {value}' not found in the data.")

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
    if selected_analyses_metric == 'General':
        # Generate insights for "General"
        data = time_series_data[selected_report_metric]
        insights = generate_insights_for_2023(data, selected_report_metric, "General")
    elif len(filtered_df.index.get_level_values(column_name).unique()) == 1:
        # Single value selected in the third filtering
        selected_value = filtered_df.index.get_level_values(column_name)[0] if column_name else selected_report_metric
        data = time_series_data[(selected_report_metric, selected_value)]
        insights = generate_insights_for_2023(data, selected_report_metric, selected_value)
    else:
        # Multiple values selected in the third filtering
        insights = []
        for value in filtered_df.index.get_level_values(column_name).unique():
            data = time_series_data[(selected_report_metric, value)]
            insights.extend(generate_insights_for_2023(data, selected_report_metric, value))

    # Limit the total number of insights to 5
    max_insights = 5
    if len(insights) > max_insights:
        insights = random.sample(insights, max_insights)

    # Display the limited insights
    for insight in insights:
        st.write(insight)
