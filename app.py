import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import datetime 


# --- 1. SETUP & LOADING ---
st.set_page_config(page_title="Airbnb Dynamic Pricing MVP", layout="wide")

@st.cache_resource
def load_model_artifacts():
    """Loads the Model and Artifacts (heavy resources)."""
    print("Loading model and artifacts...")
    model = joblib.load('model/model_latest.pkl')
    
    with open('artifacts/pricing_artifacts.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    return model, artifacts

@st.cache_data
def load_data():
    """Loads the Database (data)."""
    print("Loading data...")
    db = pd.read_parquet('data/cleaned_london_airbnb_data.parquet', engine='fastparquet')
    return db

try:
    model, artifacts = load_model_artifacts()
    db = load_data()
except FileNotFoundError as e:
    st.error(f"Critical Error: Missing file: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error loading system: {e}")
    st.stop()

# --- 2. HELPER FUNCTIONS (The Logic) ---

def find_optimal_price(listing_row, model, artifacts, min_price_mult=0.5, max_price_mult=2.0):
    """Tests 100 prices to find the winner."""
    # Clean input
    listing_row = listing_row.drop(labels=['is_booked', 'listing_id'], errors='ignore')
    
    # Base price in real £ (must reverse log1p)
    current_real_price = np.expm1(listing_row['log_price'])
    
    # Generate Test Prices (Real £)
    test_prices = np.linspace(
        start=max(20, current_real_price * min_price_mult),
        stop=current_real_price * max_price_mult, 
        num=100
    )
    
    # Convert test prices to Log for the model
    log_test_prices = np.log1p(test_prices)
    
    # Create Batch for prediction
    batch_df = pd.DataFrame([listing_row] * len(test_prices))
    batch_df['log_price'] = log_test_prices
    
    # RECALCULATE LOG PRICE COMPETITIVENESS
    # Logic: Log(Price / Avg) = Log(Price) - Log(Avg)
    neighborhood = listing_row['neighbourhood_cleansed']
    avg_log_price = artifacts['avg_log_price_lookup'].get(neighborhood, artifacts['global_avg_log_price'])
    
    batch_df['log_price_competitiveness'] = log_test_prices - avg_log_price
        
    # Predict
    valid_cols = model.feature_name_
    batch_df = batch_df.reindex(columns=valid_cols, fill_value=0)
    
    # --- FIX: Restore Categorical Types for LightGBM ---
    # LightGBM requires categorical columns to have the 'category' dtype
    cat_features = [
        'month', 'day_of_week', 'neighbourhood_cleansed', 
        'property_type', 'property_group', 'rating_binned'
    ]
    for col in cat_features:
        if col in batch_df.columns:
            batch_df[col] = batch_df[col].astype('category')
    
    probs = model.predict_proba(batch_df)[:, 1]
    expected_revenues = test_prices * probs
    
    # Find Winner
    best_idx = np.argmax(expected_revenues)
    
    return {
        'optimal_price': test_prices[best_idx],
        'max_revenue': expected_revenues[best_idx],
        'current_price': current_real_price
    }

def generate_schedule(listing_id, model, db, artifacts, days=30):
    """Generates the 30-day forecast with dates."""
    
    # A. LOOKUP
    listing_data = db[db['listing_id'] == listing_id]
    
    if listing_data.empty:
        return None 
        
    base_row = listing_data.iloc[0]
    neighborhood = base_row['neighbourhood_cleansed']
    
    # --- DEFINE START DATE ---
    start_date = datetime.date.today() + datetime.timedelta(days=1)
    
    schedule = []
    
    for day in range(days):
        sim_row = base_row.copy()
        current_date = start_date + datetime.timedelta(days=day)
        
        # 1. Update Time Features
        sim_row['lead_time'] = day + 1
        sim_row['month'] = current_date.month
        sim_row['day_of_week'] = current_date.weekday()
        sim_row['is_weekend'] = 1 if sim_row['day_of_week'] >= 5 else 0
        
        # 2. Inject Artifacts (The "Brain")
        # interaction_neigh_dow
        dow_score = artifacts['neigh_dow_lookup'].get((neighborhood, sim_row['day_of_week']), artifacts['global_mean'])
        sim_row['interaction_neigh_dow'] = dow_score
        
        # interaction_neigh_lead
        if sim_row['lead_time'] <= 3: bucket = 'LastMinute'
        elif sim_row['lead_time'] <= 7: bucket = 'ThisWeek'
        elif sim_row['lead_time'] <= 14: bucket = 'NextWeek'
        elif sim_row['lead_time'] <= 30: bucket = 'ThisMonth'
        else: bucket = 'FarOut'
        
        lead_score = artifacts['neigh_lead_lookup'].get((neighborhood, bucket), artifacts['global_mean'])
        sim_row['interaction_neigh_lead'] = lead_score
        
        # log_price_competitiveness
        avg_log_price = artifacts['avg_log_price_lookup'].get(neighborhood, artifacts['global_avg_log_price'])
        sim_row['log_price_competitiveness'] = sim_row['log_price'] - avg_log_price
        
        # 3. Optimize
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
    """Styles the dataframe with green bars for uplift and clean formatting."""
    # 1. Define columns to display
    show_cols = [
        'Date', 'Weekday', 'Current Price', 
        'Suggested Price', 'Expected Revenue', 'Revenue Uplift'
    ]
    
    # 2. Styling
    return df[show_cols].style \
        .format({
            'Current Price': '£{:,.2f}',
            'Suggested Price': '£{:,.2f}',
            'Expected Revenue': '£{:,.2f}',
            'Revenue Uplift': '+£{:,.2f}'
        }) \
        .background_gradient(
            subset=['Suggested Price'], 
            cmap='Blues', 
            low=0.4, high=0.4
        ) \
        .bar(
            subset=['Revenue Uplift'], 
            align='mid', 
            color=['#d65f5f', '#5fba7d'],
            width=90
        ) \
        .set_properties(**{'text-align': 'center'})

# --- 3. THE UI (Frontend) ---

st.title("🏨 London Airbnb Dynamic Pricing Engine")
st.markdown("Optimize your Airbnb listing using AI-driven demand forecasting.")

# Sidebar Inputs
st.sidebar.header("Configuration")
input_id = st.sidebar.number_input("Enter Listing ID", step=1, value=int(db['listing_id'].iloc[0]))
days_forecast = st.sidebar.slider("Forecast Range", 7, 60, 30)

if st.sidebar.button("🚀 Generate Prices"):
    with st.spinner(f"Analyzing market data for Listing {input_id}..."):
        
        # Run the Engine
        df_results = generate_schedule(input_id, model, db, artifacts, days_forecast)

        if df_results is not None:
            # 1. Summary Metrics
            avg_suggested = df_results['Suggested Price'].mean()
            avg_current = df_results['Current Price'].mean()
            uplift = ((avg_suggested - avg_current) / avg_current) * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg Suggested Price", f"£{avg_suggested:.0f}")
            col2.metric("Avg Current Price", f"£{avg_current:.0f}")
            col3.metric("Potential Revenue Uplift", f"{uplift:.1f}%", delta_color="normal")

            # 2. CALCULATE UPLIFT & BASELINE
            # Create a dedicated column for Baseline for plotted comparison
            df_results['Baseline Revenue'] = df_results['Current Price'] * 0.4 
            df_results['Revenue Uplift'] = df_results['Expected Revenue'] - df_results['Baseline Revenue']

            # 3. DISPLAY TABLE
            st.subheader("📅 Optimized Schedule")
            with st.container(height=600):
                st.table(display_styled_rate_card(df_results))
            
            # --- PRICE TREND CHART ---
            st.subheader("📈 Price Trend Forecast")
            price_chart_data = df_results.set_index('Date')[['Suggested Price', 'Current Price']]
            st.line_chart(
                price_chart_data, 
                color=["#00CC96", "#d3d3d3"],
                height=300
            )

            # --- REVENUE TREND CHART ---
            st.subheader("💷 Revenue Trend Forecast")
            rev_chart_data = df_results.set_index('Date')[['Baseline Revenue', 'Expected Revenue']]
            st.line_chart(
                rev_chart_data,
                color=["#d3d3d3", "#00CC96"],  
                height=300
            )
            
        else:
            st.error(f"Listing ID {input_id} not found in database.")
            st.info("Try using one of these valid IDs from your dataset:")
            st.write(db['listing_id'].unique())
