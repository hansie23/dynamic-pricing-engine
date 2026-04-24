---
title: London Airbnb Dynamic Pricing
emoji: 🏡
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# 🏨 London Airbnb Dynamic Pricing Engine

**[LIVE MVP](https://hansie23-dynamic-pricing-engine.hf.space)**  
An AI-powered tool designed for London Airbnb hosts to maximize revenue through data-driven price optimization. This engine leverages historical data and machine learning to predict booking probabilities and suggest optimal daily prices for a 30-day forecast.

---


## ✨ Key Features

*   **Listing Lookup:** Instantly fetch and analyze data for specific London Airbnb listings.
*   **AI-Driven Demand Forecasting:** Uses a LightGBM model to predict the probability of a booking based on lead time, seasonality, and neighborhood trends.
*   **Price Optimization:** Simulates 100 different price points for every day to identify the rate that maximizes expected revenue.
*   **Interactive 30-Day Schedule:** Provides a detailed table of suggested prices, expected revenue, and potential uplift.
*   **Visual Analytics:** Real-time charts for price trends and revenue forecasts compared to baseline pricing.

---

## 🛠️ Tech Stack

*   **Frontend:** Streamlit
*   **Data Processing:** Pandas, NumPy, FastParquet
*   **Machine Learning:** LightGBM, Scikit-learn, Joblib
*   **Visualization:** Matplotlib

---

## 🧠 How It Works

The engine calculates the **Expected Revenue** for each day using the following formula:
`Expected Revenue = Price * Probability of Booking`

For every day in the 30-day forecast, the system:
1.  Updates features like `lead_time`, `is_weekend`, and `month`.
2.  Retrieves neighborhood-specific historical performance scores.
3.  Tests 100 different price points (ranging from 50% to 200% of the current price).
4.  Predicts the booking probability for each price point.
5.  Selects the price that yields the highest expected revenue.
