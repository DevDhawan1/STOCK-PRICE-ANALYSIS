import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Helper function to generate dummy stock data (historical + simulated prediction) ---
def generate_dummy_data(symbol, start_date, end_date, initial_price, volatility_factor, trend_factor, prediction_years):
    """
    Generates dummy stock data including Open, High, Low, Close, Adj Close, Volume,
    and simulated future predictions.

    Args:
        symbol (str): Stock symbol.
        start_date (datetime): Start date for historical data generation.
        end_date (datetime): End date for historical data generation.
        initial_price (float): Starting price.
        volatility_factor (float): Factor for price fluctuations.
        trend_factor (float): Factor to simulate an upward trend.
        prediction_years (int): Number of years to predict into the future.

    Returns:
        pd.DataFrame: DataFrame with stock data, including 'PredictedClose' for future dates.
    """
    data = []
    current_date = start_date
    current_open = initial_price
    current_close = initial_price

    # Generate historical data
    while current_date <= end_date:
        day_vol = (np.random.rand() - 0.5) * volatility_factor * 0.01
        current_close = current_open + current_open * trend_factor + day_vol
        current_close = max(current_close, initial_price * 0.5) # Prevent price from dropping too low

        high = round(current_close * (1 + np.random.rand() * 0.02), 2) # High is slightly above close
        low = round(current_close * (1 - np.random.rand() * 0.02), 2)  # Low is slightly below close
        volume = int(1000000 + np.random.rand() * 10000000) # Random volume

        data.append({
            'Date': current_date.strftime('%Y-%m-%d'),
            'Open': round(current_open, 2),
            'High': high,
            'Low': low,
            'Close': round(current_close, 2),
            'Adj Close': round(current_close, 2), # For simplicity, Adj Close = Close
            'Volume': volume,
            'Type': 'Actual' # Mark as actual data
        })

        current_open = current_close # Next day's open is current day's close
        current_date += timedelta(days=1)

    df_historical = pd.DataFrame(data)

    # Simulate future prediction data
    last_historical_date = datetime.strptime(df_historical['Date'].iloc[-1], '%Y-%m-%d')
    future_end_date = last_historical_date + timedelta(days=int(prediction_years * 365.25))

    predicted_data = []
    current_predicted_price = df_historical['Close'].iloc[-1]
    future_date = last_historical_date + timedelta(days=1) # Start prediction from the next day

    while future_date <= future_end_date:
        # Simulate future trend with some noise
        current_predicted_price += current_predicted_price * (trend_factor * 1.5) + (np.random.rand() - 0.5) * (volatility_factor * 0.005)
        current_predicted_price = max(current_predicted_price, initial_price * 0.5) # Prevent negative prices

        predicted_data.append({
            'Date': future_date.strftime('%Y-%m-%d'),
            'Open': None, 'High': None, 'Low': None, 'Volume': None, 'Adj Close': None, # No detailed future data
            'Close': round(current_predicted_price, 2), # This will be the predicted close
            'Type': 'Predicted' # Mark as predicted data
        })
        future_date += timedelta(days=1)

    df_predicted = pd.DataFrame(predicted_data)

    return pd.concat([df_historical, df_predicted], ignore_index=True)

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

# --- Data Generation ---
# Define historical period (mimicking the notebook's 2015-2025)
historical_start_date = datetime(2015, 6, 4)
historical_end_date = datetime(2025, 6, 4)

@st.cache_data
def get_stock_data(symbol, start_date, end_date, pred_years):
    """Caches data generation to avoid re-running on every interaction."""
    # Parameters for dummy data generation based on symbol
    params = {
        'AAPL': (25, 200, 0.001),
        'TSLA': (40, 800, 0.003),
        'NFLX': (10, 400, 0.002),
        'JPM': (60, 200, 0.0005)
    }
    initial_price, volatility_factor, trend_factor = params.get(symbol, (50, 500, 0.001))
    return generate_dummy_data(symbol, start_date, end_date, initial_price, volatility_factor, trend_factor, pred_years)

st.write("Loading data... done!") # Streamlit equivalent of "Loading data... done!"

df_stock = get_stock_data(selected_symbol, historical_start_date, historical_end_date, prediction_years)

# Separate historical and predicted data
df_historical = df_stock[df_stock['Type'] == 'Actual'].copy()
df_predicted = df_stock[df_stock['Type'] == 'Predicted'].copy()

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

# Trace for Predicted Close prices
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
    margin=dict(l=50, r=50, b=80, t=100), # Increased top margin
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
    ),
    annotations=[
        go.layout.Annotation(
            xref="paper", yref="paper", x=0.95, y=0.05,
            text='Forecasted data is simulated',
            showarrow=False,
            font=dict(color='#cbd5e0', size=10),
            align='right',
            bgcolor='rgba(0,0,0,0)',
            opacity=0.7
        )
    ]
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
    margin=dict(l=50, r=50, b=80, t=100), # Increased top margin
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
    margin=dict(l=50, r=50, b=80, t=100), # Increased top margin
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
