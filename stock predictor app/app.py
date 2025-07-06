import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
import time # Import time for delays
import streamlit.components.v1 as components

# Replace G-XXXXXXXXXX with your actual GA4 Measurement ID
GA_JS = """
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-SWEFHRLPBR"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-SWEFHRLPBR');
</script>
"""

# Inject the tracking script
components.html(GA_JS, height=0)

# --- Data Loading Functions ---

# Function to load data from CSV
def load_data_from_csv(symbol):
    """
    Attempts to load historical stock data from a CSV file with a specific,
    unusual header and data structure (as observed in provided images).
    Assumes CSV has a header in row 0, a junk row in row 1, and data starting row 2 (index 2).
    Manually assigns column names.
    """
    csv_file_path = f"{symbol}_data.csv"
    try:
        st.info(f"Attempting to load data from {csv_file_path} with custom parsing...")
        
        # Read CSV: skip the first two header/junk rows and don't assume a header
        # The 'names' argument will assign these column names directly to the data.
        # Based on image_1f8665.png, the order of data columns is Date, Close, High, Low, Open, Volume
        column_names = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        df = pd.read_csv(csv_file_path, header=None, skiprows=[0, 1], names=column_names)
        
        # Clean column names (strip whitespace - although manually assigned, good practice)
        df.columns = df.columns.str.strip()

        # Check for essential columns manually assigned
        essential_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'] # Adj Close will be derived
        for col in essential_cols:
            if col not in df.columns:
                st.error(f"Missing essential column '{col}' even after custom parsing for {csv_file_path}. Data might be malformed or header structure is different. Skipping CSV loading.")
                return pd.DataFrame()

        # Ensure 'Date' column is in correct format, coercing errors
        try:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%d-%m-%Y') # Explicit format for safety
            df = df.dropna(subset=['Date']) 
            
            if df['Date'].empty:
                st.error(f"After parsing, the 'Date' column in {csv_file_path} is empty or contains no valid dates. Please ensure date format is consistent (e.g., DD-MM-YYYY). Skipping CSV loading.")
                return pd.DataFrame()

            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d') # Format back to string as YYYY-MM-DD
        except Exception as e:
            st.error(f"Error processing 'Date' column in {csv_file_path}: {e}. Please ensure date format is consistent (e.g., DD-MM-YYYY). Skipping CSV loading.")
            return pd.DataFrame()
        
        # Ensure required numeric columns are numeric, coercing errors
        numeric_cols_to_check = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols_to_check:
            df[col] = pd.to_numeric(df[col], errors='coerce') # Coerce errors to NaN
        
        # Drop rows with NaN in critical financial columns after coercion
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])

        if df.empty:
            st.warning(f"CSV file {csv_file_path} loaded but contained no valid data rows after cleaning (e.g., missing essential values or unparseable dates).")
            return pd.DataFrame()

        # Add 'Adj Close' column, assuming it's the same as 'Close' if not explicitly in CSV
        # If your yfinance export has 'Adj Close' as a separate column, you'd add it to 'column_names' list above
        if 'Adj Close' not in df.columns:
            df['Adj Close'] = df['Close'] # For consistency with yfinance output format

        df['Type'] = 'Actual'
        st.success(f"Successfully loaded {len(df)} rows from {csv_file_path}.")
        return df

    except FileNotFoundError:
        st.error(f"Fatal Error: CSV file not found for {symbol} at {csv_file_path}. Please ensure your CSV files are correctly placed in the app directory. The app cannot proceed without data.")
        return pd.DataFrame() 
    except Exception as e:
        st.error(f"Fatal Error: An unexpected error occurred while reading CSV file {csv_file_path}: {e}. Please ensure your CSV file structure matches the expected format. The app cannot proceed without data.")
        return pd.DataFrame()


@st.cache_data(ttl=3600) # Cache data for 1 hour
def get_stock_data(symbol, historical_start_date, historical_end_date, prediction_years):
    """
    Attempts to fetch historical stock data from a CSV file.
    If CSV loading fails, returns empty DataFrames and the app will stop.
    """
    df_historical = pd.DataFrame()
    
    # --- Directly try loading from CSV ---
    df_historical = load_data_from_csv(symbol)
    
    # --- If CSV loading fails, return empty DataFrames ---
    if df_historical.empty:
        # load_data_from_csv already prints the error message
        return pd.DataFrame(), pd.DataFrame() # Return empty if no data can be loaded

    # --- Simulate Future Prediction Data (always based on the successfully loaded historical data) ---
    # Ensure df_historical is sorted by date before taking the last element
    df_historical = df_historical.sort_values(by='Date').reset_index(drop=True)
    
    last_historical_date_str = df_historical['Date'].iloc[-1]
    # Ensure last_historical_close_price is a single numeric value
    last_historical_close_price = df_historical['Close'].iloc[-1]
    if isinstance(last_historical_close_price, pd.Series):
        last_historical_close_price = last_historical_close_price.iloc[0] # Take first element if it's a Series

    last_historical_date = datetime.strptime(last_historical_date_str, '%Y-%m-%d')
    future_end_date = last_historical_date + timedelta(days=int(prediction_years * 365.25))

    predicted_data = []
    current_predicted_price = last_historical_close_price
    future_date = last_historical_date + timedelta(days=1)

    trend_factor_pred = {
        'AAPL': 0.0005, 'TSLA': 0.0015, 'NFLX': 0.001, 'JPM': 0.0002
    }.get(symbol, 0.0005) 
    volatility_factor_pred = {
        'AAPL': 5, 'TSLA': 20, 'NFLX': 10, 'JPM': 2
    }.get(symbol, 5) 

    while future_date <= future_end_date:
        current_predicted_price += current_predicted_price * trend_factor_pred + (np.random.rand() - 0.5) * volatility_factor_pred
        current_predicted_price = max(current_predicted_price, 0.01) 

        predicted_data.append({
            'Date': future_date.strftime('%Y-%m-%d'),
            'Open': None, 'High': None, 'Low': None, 'Volume': None, 'Adj Close': None,
            'Close': round(current_predicted_price, 2),
            'Type': 'Predicted'
        })
        future_date += timedelta(days=1)

    df_predicted = pd.DataFrame(predicted_data)

    return df_historical, df_predicted


# --- Streamlit App Layout ---

st.set_page_config(layout="wide", page_title="Stock Forecast App")

st.title("Stock Forecast App")

# --- Controls Section ---
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Select dataset for prediction:")
    selected_symbol = st.selectbox(
        "Company",
        ('AAPL', 'TSLA', 'NFLX', 'JPM'),
        index=1, # Default to Tesla
        label_visibility="collapsed"
    )

with col2:
    st.markdown("### Years of prediction:")
    prediction_years = st.slider(
        "Prediction Years",
        min_value=1,
        max_value=4,
        value=2,
        step=1,
        label_visibility="collapsed"
    )

# --- Data Fetching and Separation ---
# Define historical period (mimicking the notebook's 2015-2025)
historical_start_date = datetime(2015, 6, 4)
historical_end_date = datetime.today() # This is now just for defining the prediction end point, not for fetching historical

st.write("Loading data... (Attempting to load from CSV files.)")
df_historical, df_predicted = get_stock_data(selected_symbol, historical_start_date, historical_end_date, prediction_years)

if df_historical.empty and df_predicted.empty:
    # Error message is already displayed by load_data_from_csv or get_stock_data
    st.stop() # Stop execution if no data at all

st.write("Loading data... done!")

# --- Combined Plot: Stock Price (Actual & Forecast) ---
st.markdown("### Stock Price (Actual & Forecast)")

fig_combined = go.Figure()

# Trace for Actual Open prices
fig_combined.add_trace(go.Scatter(
    x=df_historical['Date'],
    y=df_historical['Open'],
    mode='lines',
    name='Actual Open',
    line=dict(color='#a0aec0', width=1),
    hovertemplate='<b>Date:</b> %{x}<br><b>Open:</b> %{y:.2f}<extra></extra>'
))

# Trace for Actual Close prices
fig_combined.add_trace(go.Scatter(
    x=df_historical['Date'],
    y=df_historical['Close'],
    mode='lines',
    name='Actual Close',
    line=dict(color='#63b3ed', width=2),
    hovertemplate='<b>Date:</b> %{x}<br><b>Close:</b> %{y:.2f}<extra></extra>'
))

# Trace for Predicted Close prices (only if predicted data exists)
if not df_predicted.empty:
    fig_combined.add_trace(go.Scatter(
        x=df_predicted['Date'],
        y=df_predicted['Close'],
        mode='lines',
        name='Predicted Close',
        line=dict(color='#f6ad55', width=2, dash='dot'),
        hovertemplate='<b>Date:</b> %{x}<br><b>Predicted:</b> %{y:.2f}<extra></extra>'
    ))

fig_combined.update_layout(
    title={
        'text': f"{selected_symbol} Stock Price (Actual & Forecast)",
        'font': {'family': 'Rockwell, sans-serif', 'size': 26, 'color': 'white'},
        'x': 0.5, # Centered title
        'xanchor': 'center' # Centered title
    },
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    xaxis=dict(
        showline=True, showgrid=True, showticklabels=True,
        linecolor='white', linewidth=2, ticks='outside',
        tickfont=dict(family='Rockwell', size=12, color='white'),
        rangeslider=dict(visible=True, bgcolor='#2d3748', thickness=0.07),
        type='date'
    ),
    yaxis=dict(
        showline=True, showgrid=True, showticklabels=True,
        linecolor='white', linewidth=2, ticks='outside',
        tickfont=dict(family='Rockwell', size=12, color='white')
    ),
    paper_bgcolor='rgba(0,0,0,0)', # Transparent background for the plot area
    plot_bgcolor='rgba(0,0,0,0)', # Transparent background for the chart area
    autosize=True,
    margin=dict(l=50, r=50, b=80, t=100), # Increased top margin for title + legend
    hovermode='x unified',
    template='plotly_dark',
    legend=dict(
        x=0.5, y=1.0, # Centered below the plot title, within increased top margin
        xanchor='center',
        yanchor='bottom', # Anchor to bottom of legend
        bgcolor='rgba(0,0,0,0)',
        bordercolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        orientation='h'
    )
)

st.plotly_chart(fig_combined, use_container_width=True)

# --- Historical Open/Close Price Plot ---
st.markdown("### Historical Stock Open and Close Prices")

fig_historical = go.Figure()

fig_historical.add_trace(go.Scatter(
    x=df_historical['Date'],
    y=df_historical['Open'],
    mode='lines',
    name='Open Price',
    line=dict(color='#ADD8E6', width=2), # Light blue for open
    hovertemplate='<b>Date:</b> %{x}<br><b>Open:</b> %{y:.2f}<extra></extra>'
))

fig_historical.add_trace(go.Scatter(
    x=df_historical['Date'],
    y=df_historical['Close'],
    mode='lines',
    name='Close Price',
    line=dict(color='#90EE90', width=2), # Light green for close
    hovertemplate='<b>Date:</b> %{x}<br><b>Close:</b> %{y:.2f}<extra></extra>'
))

fig_historical.update_layout(
    title={
        'text': f"{selected_symbol} Historical Open and Close Prices",
        'font': {'family': 'Rockwell', 'size': 24, 'color': 'white'},
        'x': 0.5, 'xanchor': 'center' # Centered title
    },
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    xaxis=dict(
        showline=True, showgrid=True, showticklabels=True,
        linecolor='white', linewidth=2, ticks='outside',
        tickfont=dict(family='Rockwell', size=12, color='white'),
        rangeslider=dict(visible=True, bgcolor='#2d3748', thickness=0.07),
        type='date'
    ),
    yaxis=dict(
        showline=True, showgrid=True, showticklabels=True,
        linecolor='white', linewidth=2, ticks='outside',
        tickfont=dict(family='Rockwell', size=12, color='white')
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    autosize=True,
    margin=dict(l=50, r=50, b=80, t=100), # Increased top margin for title + legend
    hovermode='x unified',
    template='plotly_dark',
    legend=dict(
        x=0.5, y=1.0, # Centered below the plot title, within increased top margin
        xanchor='center',
        yanchor='bottom', # Anchor to bottom of legend
        bgcolor='rgba(0,0,0,0)',
        bordercolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        orientation='h'
    )
)

st.plotly_chart(fig_historical, use_container_width=True)

# --- Raw Data Table ---
st.markdown("### Raw Data")

# Display only the last 100 historical entries for performance and readability
recent_historical_data = df_historical.tail(100).sort_values(by='Date', ascending=False)
st.dataframe(
    recent_historical_data[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].style.format(
        {
            'Open': "{:.2f}",
            'High': "{:.2f}",
            'Low': "{:.2f}",
            'Close': "{:.2f}",
            'Adj Close': "{:.2f}",
            'Volume': "{:,}"
        }
    ).set_table_styles(
        [
            {'selector': 'thead', 'props': [('background-color', '#4a5568'), ('color', 'white')]},
            {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#2d3748')]},
            {'selector': 'tbody tr:nth-child(odd)', 'props': [('background-color', '#1a202c')]},
            {'selector': 'td, th', 'props': [('border', '1px solid #4a5568')]},
            {'selector': 'td', 'props': [('color', '#e2e8f0')]}
        ]
    ),
    hide_index=True,
    use_container_width=True,
    height=400 # Fixed height for the table
)


# --- Forecast Plot ---
st.markdown("### Forecast Plot")

fig_forecast = go.Figure()

# Trace for Predicted Close prices (only if predicted data exists)
if not df_predicted.empty:
    fig_forecast.add_trace(go.Scatter(
        x=df_predicted['Date'],
        y=df_predicted['Close'],
        mode='lines',
        name='Predicted Close Price',
        line=dict(color='#f6ad55', width=2, dash='solid'), # Solid orange for forecast
        hovertemplate='<b>Date:</b> %{x}<br><b>Predicted:</b> %{y:.2f}<extra></extra>'
    ))

fig_forecast.update_layout(
    title={
        'text': f"{selected_symbol} Forecasted Close Prices",
        'font': {'family': 'Rockwell', 'size': 24, 'color': 'white'},
        'x': 0.5, 'xanchor': 'center' # Centered title
    },
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    xaxis=dict(
        showline=True, showgrid=True, showticklabels=True,
        linecolor='white', linewidth=2, ticks='outside',
        tickfont=dict(family='Rockwell', size=12, color='white'),
        rangeslider=dict(visible=True, bgcolor='#2d3748', thickness=0.07),
        type='date'
    ),
    yaxis=dict(
        showline=True, showgrid=True, showticklabels=True,
        linecolor='white', linewidth=2, ticks='outside',
        tickfont=dict(family='Rockwell', size=12, color='white')
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    autosize=True,
    margin=dict(l=50, r=50, b=80, t=100), # Increased top margin for title + legend
    hovermode='x unified',
    template='plotly_dark',
    legend=dict(
        x=0.5, y=1.0, # Centered below the plot title, within increased top margin
        xanchor='center',
        yanchor='bottom', # Anchor to bottom of legend
        bgcolor='rgba(0,0,0,0)',
        bordercolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        orientation='h'
    )
)

st.plotly_chart(fig_forecast, use_container_width=True)

# --- Forecast Data Table ---
st.markdown("### Forecast Data Table")

# Display only the first 100 predicted entries for performance and readability
recent_predicted_data = df_predicted.head(100) # Show first 100 forecasted entries
st.dataframe(
    recent_predicted_data[['Date', 'Close']].rename(columns={'Close': 'Predicted Close'}).style.format(
        {
            'Predicted Close': "{:.2f}"
        }
    ).set_table_styles(
        [
            {'selector': 'thead', 'props': [('background-color', '#4a5568'), ('color', 'white')]},
            {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#2d3748')]},
            {'selector': 'tbody tr:nth-child(odd)', 'props': [('background-color', '#1a202c')]},
            {'selector': 'td, th', 'props': [('border', '1px solid #4a5568')]},
            {'selector': 'td', 'props': [('color', '#e2e8f0')]}
        ]
    ),
    hide_index=True,
    use_container_width=True,
    height=400 # Fixed height for the table
)
