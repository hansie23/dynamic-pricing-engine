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

def find_optimal_price(listing_row, model, min_price_mult=0.5, max_price_mult=2.0):
    """Tests 100 prices to find the winner."""
    # Clean input
    listing_row = listing_row.drop(labels=['is_booked', 'listing_id'], errors='ignore')
    current_price = listing_row['price']
    
    # Generate Test Prices
    test_prices = np.linspace(
        start=max(20, current_price * min_price_mult),
        stop=current_price * max_price_mult, 
        num=100
    )
    
    # Create Batch
    batch_df = pd.DataFrame([listing_row] * len(test_prices))
    batch_df['price'] = test_prices
    
    # RECALCULATE RELATIVE PRICE (Crucial!)
    if 'price_competitiveness' in listing_row:
        # Reverse engineer the avg price from the current row
        avg_neigh_price = current_price / listing_row['price_competitiveness']
        if avg_neigh_price == 0: avg_neigh_price = 1
        batch_df['price_competitiveness'] = batch_df['price'] / avg_neigh_price
        
    # Predict
    # Keep only valid columns
    valid_cols = model.feature_name_
    
    # Ensure columns match
    batch_df.columns = [str(c).replace(' ', '_').replace('/', '_') for c in batch_df.columns]
    batch_df = batch_df.reindex(columns=valid_cols, fill_value=0)
    
    probs = model.predict_proba(batch_df)[:, 1]
    expected_revenues = test_prices * probs
    
    # Find Winner
    best_idx = np.argmax(expected_revenues)
    
    return {
        'optimal_price': test_prices[best_idx],
        'max_revenue': expected_revenues[best_idx],
        'current_price': current_price
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
    # Start predicting from "Tomorrow" relative to today
    start_date = datetime.date.today() + datetime.timedelta(days=1)
    
    schedule = []
    
    for day in range(days): # Loop 0 to 29
        sim_row = base_row.copy()
        
        # --- CALCULATE ACTUAL DATE ---
        current_date = start_date + datetime.timedelta(days=day)
        
        # 1. Update Time Features Correctly
        sim_row['lead_time'] = day + 1          # Lead time starts at 1 (Tomorrow)
        sim_row['month'] = current_date.month   # Uses the REAL month (e.g., 12 for Dec)
        sim_row['day_of_week'] = current_date.weekday() # 0=Mon, 6=Sun
        sim_row['day_of_year'] = current_date.timetuple().tm_yday # 1-365
        sim_row['is_weekend'] = 1 if sim_row['day_of_week'] >= 5 else 0
        
        # 2. Inject Artifacts (The "Brain")
        # Neigh x Day
        dow_score = artifacts['neigh_dow_lookup'].get((neighborhood, sim_row['day_of_week']), artifacts['global_mean'])
        sim_row['interaction_neigh_dow'] = dow_score
        
        # Neigh x Lead Time
        if sim_row['lead_time'] <= 3: bucket = 'LastMinute'
        elif sim_row['lead_time'] <= 7: bucket = 'ThisWeek'
        elif sim_row['lead_time'] <= 14: bucket = 'NextWeek'
        elif sim_row['lead_time'] <= 30: bucket = 'ThisMonth'
        else: bucket = 'FarOut'
        
        lead_score = artifacts['neigh_lead_lookup'].get((neighborhood, bucket), artifacts['global_mean'])
        sim_row['interaction_neigh_lead'] = lead_score
        
        # Relative Price (using the artifact)
        avg_neigh_price = artifacts['avg_price_lookup'].get(neighborhood, artifacts['global_avg_price'])
        if avg_neigh_price == 0: avg_neigh_price = 1
        sim_row['price_competitiveness'] = sim_row['price'] / avg_neigh_price
        
        # 3. Optimize
        res = find_optimal_price(sim_row, model)
        
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
