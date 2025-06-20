# 📈 Stock Market Forecasting Using Deep Learning

This project delves into forecasting stock prices of major companies, initially leveraging advanced deep learning models like **LSTM** and **GRU**, and extending into an interactive **web application** for data visualization and simulated future trends. By learning from historical patterns, the project aims to predict future closing prices and provide insightful visualizations of market behavior.

It encompasses a full data science pipeline: **data collection, robust pre-processing, statistical analysis, time series forecasting with deep learning, and interactive deployment**.

---

## 📊 Stocks Analyzed

- **JPM** — JPMorgan Chase & Co.  
- **NFLX** — Netflix, Inc.  
- **AAPL** — Apple Inc.  
- **TSLA** — Tesla Inc.  

**📅 Historical Data Date Range:** 4 June 2015 – 4 June 2025  
**🛠️ Interval:** Daily  

---

## 🧠 Technologies Used

- **Python 3.13**
- **Jupyter Notebook** (for research, model training, and analysis)
- **Streamlit** (for interactive web app)

**Libraries:**
- `pandas`, `numpy` – Data manipulation and numerical operations  
- `matplotlib`, `seaborn` – Static visualizations  
- `plotly` – Interactive visualizations  
- `scikit-learn` – Preprocessing (MinMaxScaler)  
- `keras` (TensorFlow backend) – Deep learning model training  
- `yfinance` – Historical data fetching  

---

## 📥 Data Source

- All stock data during model training was fetched using the `yfinance` Python package.
- For deployment, historical data is read from pre-compiled CSV files stored within the project repository to ensure fast, reliable app performance.

---

## 🏗️ Project Workflow

### 1. **Data Collection**
- Used `yfinance` to fetch historical data (2015–2025) for each ticker.

### 2. **Preprocessing**
- Selected `'Close'` prices for modeling.
- Normalized data with **MinMaxScaler**.
- Created **100-day lookback windows** for time series sequences.

### 3. **Model Architecture**
- **LSTM**: Captures long-term dependencies.
- **GRU**: Efficient alternative with similar capabilities.

Each stock was modeled independently.

### 4. **Training & Testing**
- 80% training / 20% testing split.
- Models trained for **100 epochs** (configurable).

### 5. **Visualization**
- Plotted **actual vs. predicted prices** to evaluate accuracy and trend alignment.

---

## 🌐 Interactive Web Application

Built using **Streamlit** to enable user interaction and forecast exploration.

### 🔗 [Live Demo](https://stock-predictor-devdhawan.streamlit.app/)


### 🔧 Features

- **Dynamic Stock Selection**: Choose AAPL, TSLA, NFLX, or JPM.
- **Forecast Period Slider**: Customize between 1 to 4 future years.
- **Interactive Visualizations**:
  - Actual & forecasted prices (Plotly)
  - Open/Close price history
  - Simulated future forecast
- **Data Tables**:
  - Historical OHLCV data
  - Simulated forecast data

### ⚠️ Forecasts in the App

> Forecasts are currently **simulated** using statistical trends from the data (not deep learning model outputs). Integration of trained LSTM/GRU predictions is planned for future updates.

---

## 🔎 Insights from Jupyter Notebook

### 📈 High Value Trends

- **AAPL**: Steady, linear post-2019 growth.
- **TSLA**: Exponential rise after 2019 with high volatility.
- **NFLX**: Growth between 2018–2021, minor dips after.
- **JPM**: Stable price behavior, easier to predict.

### 📉 Limitations

- Sudden jumps or dips (TSLA, NFLX) caused prediction lag.
- Only using `'Close'` prices limits model understanding of market context.

---

## 🧪 Model Evaluation

### ✅ Test Set Forecast Performance

| Ticker | Performance       | Notes                                         |
|--------|-------------------|-----------------------------------------------|
| AAPL   | ✅ High Accuracy   | Smooth trend, minimal lag                     |
| TSLA   | ⚠️ Moderate       | Volatility led to slight underfitting         |
| NFLX   | ✅ Good            | Missed a few sharp dips                       |
| JPM    | ✅ Stable          | Low volatility ensured strong prediction fit  |

### 📊 Visual Output

- **Blue Line**: Actual Closing Prices  
- **Orange Line**: Predicted Prices by LSTM  

---

## 📸 Sample Model Visualizations

### 🍏 Apple Inc. (AAPL)
- Accurate long-term prediction.
- Handles stable pattern effectively.

### 🚗 Tesla Inc. (TSLA)
- Captures overall trend.
- Struggles during sharp spikes.

### 📺 Netflix Inc. (NFLX)
- Handles seasonal patterns well.
- Misses occasional sharp dips.

### 🏦 JPMorgan Chase & Co. (JPM)
- Predictions closely follow actual prices.

---

## 🔗 Connect with Me

Feel free to reach out or explore more:

- 📇 [LinkedIn – Devansh Dhawan](https://www.linkedin.com/in/devansh-dhawan)
- 📁 [GitHub Profile](https://github.com/devanshdhawan8943)
- 📬 Email: **devanshdhawan8943@gmail.com**
- 🔗 [LinkedIn Project Post]([https://www.linkedin.com/posts/devansh-dhawan_your-post-id-here](https://www.linkedin.com/feed/update/urn:li:activity:7341809534606155777/))

---

## 🎉 Thanks for reading!

If you found this project helpful, please ⭐ star the repo, share your feedback, or connect on socials!
