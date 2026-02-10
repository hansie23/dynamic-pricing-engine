# Product Requirements Document (PRD): Airbnb Dynamic Pricing Engine

## 1. Executive Summary
The Airbnb Dynamic Pricing Engine is an AI-powered tool designed for London Airbnb hosts. It leverages historical data and machine learning to predict booking probabilities and suggest optimal daily prices that maximize expected revenue.

## 2. Problem Statement
Airbnb hosts often struggle to set prices that balance high occupancy with maximum revenue. Static pricing fails to account for seasonal demand, lead time variations, and local market competitiveness, leading to "money left on the table" or vacant listings.

## 3. Goals & Objectives
- **Primary Goal:** Increase host revenue by 10-20% through data-driven price optimization.
- **Objectives:**
  - Provide a 30-day forecast of suggested prices.
  - Visualize revenue uplift compared to baseline pricing.
  - Simplify the pricing decision process for non-technical hosts.

## 4. Target Audience
- Individual Airbnb hosts in London.
- Property management companies managing multiple short-term rentals.

## 5. Functional Requirements
### 5.1 MVP Features (Current)
- **Listing Lookup:** Ability to fetch listing data by ID.
- **Demand Forecasting:** AI model predicting booking probability.
- **Price Optimization:** Simulation of 100 price points to find the revenue-maximizing rate.
- **Dynamic Schedule:** A table showing dates, suggested prices, and expected revenue.
- **Visual Analytics:** Line charts for price and revenue trends.

### 5.2 Future Features (V1.0+)
- **Host Dashboard:** User accounts to save and monitor multiple listings.
- **Real-time Market Pulse:** Integration with live competitor pricing.
- **Seasonal Event Detection:** Adjusting prices for holidays, festivals, and major London events.
- **Direct Integration:** Automated syncing of suggested prices to Airbnb via API.
- **Multi-Listing Comparison:** Benchmarking performance across a portfolio.

## 6. Non-Functional Requirements
- **Performance:** Price optimization results should be generated in under 5 seconds.
- **Usability:** Clean, intuitive UI/UX using Streamlit/Material Design.
- **Reliability:** Accurate handling of various listing types and neighborhoods.

## 7. Success Metrics
- **Revenue Uplift:** Average % increase in expected revenue for optimized listings.
- **User Engagement:** Number of listing IDs checked per session.
- **Accuracy:** Correlation between predicted booking probability and actual booking rates (to be tracked post-implementation).
