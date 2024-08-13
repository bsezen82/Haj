import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load the final data from the CSV file
file_path = '/mnt/data/final_data.csv'
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
    'Destination Province': 'B',  # Replace 'B' with the actual column name for Destination Province
    'Origin': 'C',
    'Purpose of Visit': 'D',
    'Mode of Arrival': 'E',
    'Type of Accommodation': 'F',
    'Spending Type': 'G'
}
selected_analyses_metric = st.selectbox('Select Analyses Metric', analyses_metric_options.keys())

# Further filtering based on the Analyses Metric
column_name = analyses_metric_options[selected_analyses_metric]

# Apply filters to other columns as "All"
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

# Verify the final filtered DataFrame
st.write("Filtered DataFrame:", filtered_df)

# Option to make prediction
make_prediction = st.checkbox('Make predictions for 2024')

# Check if filtered_df is empty or missing expected indices
if not filtered_df.empty:
    # Set the index to be the combination of Report Metric and the selected Analyses Metric
    if column_name:
        filtered_df.set_index(['Report Metric', column_name], inplace=True)

    # Extract the time series data and ensure it's timezone-naive
    time_series_data = filtered_df.loc[:, '01.01.2021':].T
    time_series_data.index = pd.to_datetime(time_series_data.index, format='%d.%m.%Y', errors='coerce').tz_localize(None)

    # Ensure the selected value (e.g., 'Albaha') is in the columns
    if selected_filters and selected_filters[0] in time_series_data.columns:
        series = time_series_data[selected_filters[0]]

        # Plot the data
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(series.index, series, label=selected_filters[0])
        ax.set_xlabel('Date')
        ax.set_ylabel(selected_report_metric)
        ax.set_title(f'Trend Analysis: {selected_report_metric} in {selected_filters[0]}')
        ax.legend()
        st.pyplot(fig)
        
        # Add predictions if the checkbox is selected
        if make_prediction:
            model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=12)
            model_fit = model.fit()
            future_index = pd.date_range(start='2024-01-01', periods=12, freq='M').tz_localize(None)
            y_pred = model_fit.forecast(steps=12)
            ax.plot(future_index, y_pred, linestyle='--', label=f'2024 Prediction')
            st.pyplot(fig)
    else:
        st.error(f"Selected value '{selected_filters[0]}' not found in the time series data.")
else:
    st.error("No data available for the selected filters.")
