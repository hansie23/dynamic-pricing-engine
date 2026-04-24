import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import datetime 
import os
from huggingface_hub import hf_hub_download

# UI Setup
st.set_page_config(page_title="London Airbnb Pricing Tool", layout="wide")

# My HF Repo for the model
REPO_ID = "hansie23/dynamic-pricing-engine"

@st.cache_resource
def load_model_artifacts():
    """
    I'm loading the model and artifacts here. 
    It checks locally first for dev, then pulls from my HF Hub repo for production.
    """
    model_path = 'model/model_latest.pkl'
    artifacts_path = 'artifacts/pricing_artifacts.pkl'

    # Check for local files first
    if os.path.exists(model_path) and os.path.exists(artifacts_path):
        m = joblib.load(model_path)
        with open(artifacts_path, 'rb') as f:
            a = pickle.load(f)
        return m, a

    # Pull from Hugging Face if local files are missing (Production)
    try:
        m_hub = hf_hub_download(repo_id=REPO_ID, filename="model_latest.pkl")
        a_hub = hf_hub_download(repo_id=REPO_ID, filename="pricing_artifacts.pkl")
        
        m = joblib.load(m_hub)
        with open(a_hub, 'rb') as f:
            a = pickle.load(f)
        return m, a
    except Exception as e:
        st.error(f"Failed to load from Hub: {e}")
        st.stop()

@st.cache_data
def load_data():
    # Load the cleaned dataset
    return pd.read_parquet('data/cleaned_london_airbnb_data.parquet', engine='fastparquet')

# Initialize the system
try:
    model, artifacts = load_model_artifacts()
    db = load_data()
except Exception as e:
    st.error(f"System boot error: {e}")
    st.stop()

# --- Optimization Core ---

def find_optimal_price(listing_row, model, artifacts, min_price_mult=0.5, max_price_mult=2.0):
    """Testing 100 price points to find the sweet spot for revenue."""
    listing_row = listing_row.drop(labels=['is_booked', 'listing_id'], errors='ignore')
    
    current_real_price = np.expm1(listing_row['log_price'])
    
    # Range of prices to simulate
    test_prices = np.linspace(
        start=max(20, current_real_price * min_price_mult),
        stop=current_real_price * max_price_mult, 
        num=100
    )
    
    log_test_prices = np.log1p(test_prices)
    
    # Build the simulation batch
    batch_df = pd.DataFrame([listing_row] * len(test_prices))
    batch_df['log_price'] = log_test_prices
    
    # Recalculate competitiveness score for each simulated price
    neighborhood = listing_row['neighbourhood_cleansed']
    avg_log_price = artifacts['avg_log_price_lookup'].get(neighborhood, artifacts['global_avg_log_price'])
    batch_df['log_price_competitiveness'] = log_test_prices - avg_log_price
        
    # Ensure columns match model expectations
    valid_cols = model.feature_name_
    batch_df = batch_df.reindex(columns=valid_cols, fill_value=0)
    
    # Force categorical types for LightGBM
    cat_features = ['month', 'day_of_week', 'neighbourhood_cleansed', 'property_type', 'property_group', 'rating_binned']
    for col in cat_features:
        if col in batch_df.columns:
            batch_df[col] = batch_df[col].astype('category')
    
    probs = model.predict_proba(batch_df)[:, 1]
    expected_revenues = test_prices * probs
    
    best_idx = np.argmax(expected_revenues)
    
    return {
        'optimal_price': test_prices[best_idx],
        'max_revenue': expected_revenues[best_idx],
        'current_price': current_real_price
    }

def generate_schedule(listing_id, model, db, artifacts, days=30):
    """Building the 30-day forecast table."""
    listing_data = db[db['listing_id'] == listing_id]
    
    if listing_data.empty:
        return None 
        
    base_row = listing_data.iloc[0]
    neighborhood = base_row['neighbourhood_cleansed']
    start_date = datetime.date.today() + datetime.timedelta(days=1)
    
    schedule = []
    
    for day in range(days):
        sim_row = base_row.copy()
        current_date = start_date + datetime.timedelta(days=day)
        
        # Update time-based features for the forecast day
        sim_row['lead_time'] = day + 1
        sim_row['month'] = current_date.month
        sim_row['day_of_week'] = current_date.weekday()
        sim_row['is_weekend'] = 1 if sim_row['day_of_week'] >= 5 else 0
        
        # Pull smoothed neighborhood scores from artifacts
        sim_row['interaction_neigh_dow'] = artifacts['neigh_dow_lookup'].get((neighborhood, sim_row['day_of_week']), artifacts['global_mean'])
        
        # Handle lead time buckets
        lt = sim_row['lead_time']
        if lt <= 3: bucket = 'LastMinute'
        elif lt <= 7: bucket = 'ThisWeek'
        elif lt <= 14: bucket = 'NextWeek'
        elif lt <= 30: bucket = 'ThisMonth'
        else: bucket = 'FarOut'
        
        sim_row['interaction_neigh_lead'] = artifacts['neigh_lead_lookup'].get((neighborhood, bucket), artifacts['global_mean'])
        
        # Relative price competitiveness
        avg_log_p = artifacts['avg_log_price_lookup'].get(neighborhood, artifacts['global_avg_log_price'])
        sim_row['log_price_competitiveness'] = sim_row['log_price'] - avg_log_p
        
        res = find_optimal_price(sim_row, model, artifacts)
        
        schedule.append({
            'Day Ahead': day + 1,
            'Date': current_date.strftime('%Y-%m-%d'),
            'Weekday': current_date.strftime('%a'),
            'Current Price': res['current_price'],
            'Suggested Price': res['optimal_price'],
            'Expected Revenue': res['max_revenue']
        })
        
    return pd.DataFrame(schedule)

def display_styled_rate_card(df):
    """Custom styling for the results table."""
    show_cols = ['Date', 'Weekday', 'Current Price', 'Suggested Price', 'Expected Revenue', 'Revenue Uplift']
    return df[show_cols].style \
        .format({
            'Current Price': '£{:,.2f}',
            'Suggested Price': '£{:,.2f}',
            'Expected Revenue': '£{:,.2f}',
            'Revenue Uplift': '+£{:,.2f}'
        }) \
        .background_gradient(subset=['Suggested Price'], cmap='Blues', low=0.4, high=0.4) \
        .bar(subset=['Revenue Uplift'], align='mid', color=['#d65f5f', '#5fba7d'], width=90) \
        .set_properties(**{'text-align': 'center'})

# --- Dashboard Layout ---

st.title("London Airbnb Dynamic Pricing Engine")
st.markdown("Automated pricing suggestions based on local market demand.")

st.sidebar.header("Settings")
input_id = st.sidebar.number_input("Listing ID", step=1, value=int(db['listing_id'].iloc[0]))
days_forecast = st.sidebar.slider("Forecast Window (Days)", 7, 60, 30)

if st.sidebar.button("Run Optimizer"):
    with st.spinner(f"Simulating market demand for {input_id}..."):
        df_results = generate_schedule(input_id, model, db, artifacts, days_forecast)

        if df_results is not None:
            avg_suggested = df_results['Suggested Price'].mean()
            avg_current = df_results['Current Price'].mean()
            uplift = ((avg_suggested - avg_current) / avg_current) * 100
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Suggested Avg", f"£{avg_suggested:.0f}")
            m2.metric("Current Avg", f"£{avg_current:.0f}")
            m3.metric("Expected Uplift", f"{uplift:.1f}%")

            df_results['Baseline Revenue'] = df_results['Current Price'] * 0.4 
            df_results['Revenue Uplift'] = df_results['Expected Revenue'] - df_results['Baseline Revenue']

            st.subheader("Pricing Schedule")
            with st.container(height=600):
                st.table(display_styled_rate_card(df_results))
            
            st.subheader("Price Forecast")
            st.line_chart(df_results.set_index('Date')[['Suggested Price', 'Current Price']], color=["#00CC96", "#d3d3d3"])

            st.subheader("Revenue Forecast")
            st.line_chart(df_results.set_index('Date')[['Baseline Revenue', 'Expected Revenue']], color=["#d3d3d3", "#00CC96"])
            
        else:
            st.error(f"ID {input_id} not found. Try another one.")
